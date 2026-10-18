"""Unit checks for the Binance range importer; fixtures are not market evidence."""
from datetime import date
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from rles_data import binance_archive
from rles_data.core import Availability, DAY_NS, Event, MINUTE_NS
from rles_data.io import read_curated

pytestmark = [pytest.mark.unit, pytest.mark.micro_golden]


def test_import_range_fetches_each_day_and_month_once_and_clips_funding(tmp_path, monkeypatch):
    fetched = []

    def fake_fetch(relative_url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")
        fetched.append(relative_url)
        return {"path": str(destination), "url": "fixture:" + relative_url, "sha256": "a" * 64}

    def timestamp(day):
        epoch_ordinal = date(1970, 1, 1).toordinal()
        return (date.fromisoformat(day).toordinal() - epoch_ordinal) * DAY_NS + MINUTE_NS

    def funding_timestamp(day):
        epoch_ordinal = date(1970, 1, 1).toordinal()
        return (date.fromisoformat(day).toordinal() - epoch_ordinal) * DAY_NS

    def fake_archive_events(path, *, stream, instrument, source_url, expected_sha256, delay_ns):
        if stream == "fundingRate":
            # Include one settlement beyond the requested range to exercise clipping.
            month = path.stem.split("fundingRate-", 1)[1]
            dates = {"2024-01": ("2024-01-31",),
                     "2024-02": ("2024-02-01", "2024-02-02")}[month]
            result = []
            for i, day in enumerate(dates):
                ts = funding_timestamp(day)
                for kind, payload in (
                    ("funding_settlement", {"rate": "0.0001", "settlement_ns": ts,
                                             "interval_minutes": 480}),
                    ("funding_update", {"rate": "0.0001", "rate_semantics": "last_settled_rate"}),
                ):
                    result.append(Event(f"{month}-{kind}-{i}", kind, ts, ts + delay_ns,
                                        ts + delay_ns, Availability.DERIVED, expected_sha256,
                                        {"instrument": instrument, **payload}, source_url))
            return tuple(result)

        day = path.stem[-10:]
        ts = timestamp(day)
        if stream == "bookTicker":
            return (Event(f"{day}-best_quote", "best_quote", ts, ts + delay_ns, ts + delay_ns,
                          Availability.DERIVED, expected_sha256,
                          {"instrument": instrument, "bid_price": "100.00", "bid_qty": "0.1",
                           "ask_price": "100.01", "ask_qty": "0.2", "update_id": 1,
                           "transaction_time_ns": ts}, source_url),)
        kind = {"klines": "ohlcv_1m", "markPriceKlines": "mark",
                "indexPriceKlines": "index"}[stream]
        payload = ({"open": "100", "high": "101", "low": "99", "close": "100", "volume": "2"}
                   if kind == "ohlcv_1m" else {"price": "100"})
        return (Event(f"{day}-{kind}", kind, ts, ts + delay_ns, ts + delay_ns,
                      Availability.DERIVED, expected_sha256,
                      {"instrument": instrument, **payload}, source_url),)

    monkeypatch.setattr(binance_archive, "fetch_archive", fake_fetch)
    monkeypatch.setattr(binance_archive, "archive_events", fake_archive_events)
    output = tmp_path / "bundle"
    (output / "raw").mkdir(parents=True)
    result = binance_archive.import_range(symbol="ETHUSDT", start_day="2024-01-31",
                                          end_day="2024-02-01", output=output,
                                          delay_ns=1_000_000)

    assert len([url for url in fetched if url.startswith("daily/")]) == 6
    assert len([url for url in fetched if url.startswith("monthly/")]) == 2
    store = read_curated(output / "market.parquet")
    assert sum(event.kind == "ohlcv_1m" for event in store.events) == 2
    assert sum(event.kind == "funding_settlement" for event in store.events) == 2
    first_bar = min(event.ts_event_ns for event in store.events if event.kind == "ohlcv_1m")
    first_day_settlement = min(event.ts_event_ns for event in store.events
                                if event.kind == "funding_settlement")
    assert first_day_settlement == first_bar - MINUTE_NS
    assert result["instrument_rules_supplied"] is False
    assert result["schema"] == "rles.binance-archive-range.v1"


def test_aggtrades_archive_becomes_causal_one_second_aggressor_flow(tmp_path):
    source = tmp_path / "ETHUSDT-aggTrades-2024-01-01.zip"
    csv_bytes = (
        b"agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        b"101,2000.00,0.10,500,500,1700000000000,false\n"
        b"102,1999.00,0.20,501,502,1700000000000,true\n"
        b"103,2001.00,0.30,503,503,1700000001500,false\n"
    )
    with ZipFile(source, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("ETHUSDT-aggTrades-2024-01-01.csv", csv_bytes)
    digest = sha256(source.read_bytes()).hexdigest()

    events = binance_archive.archive_events(
        source, stream="aggTrades", instrument="ETHUSDT-PERP.SIM",
        source_url="fixture:aggTrades", expected_sha256=digest, delay_ns=1_000_000,
    )

    assert len(events) == 2
    first, second = events
    assert first.kind == second.kind == "trade_flow_1s"
    assert first.ts_event_ns == 1_700_000_001_000_000_000
    assert first.available_at_ns == first.ts_event_ns + 1_000_000
    assert first.payload["trade_count"] == 2
    assert first.payload["aggressor_buy_qty"] == "0.10"
    assert first.payload["aggressor_sell_qty"] == "0.20"
    assert first.payload["aggressor_buy_notional"] == "200.0000"
    assert first.payload["aggressor_sell_notional"] == "399.8000"
    assert first.payload["first_trade_id"] == 500
    assert first.payload["last_trade_id"] == 502
    assert first.payload["high_price"] == "2000.00"
    assert first.payload["low_price"] == "1999.00"
    assert second.ts_event_ns == 1_700_000_002_000_000_000
    assert second.payload["trade_count"] == 1


def test_aggtrades_archive_rejects_out_of_order_trade_ids(tmp_path):
    source = tmp_path / "bad.zip"
    csv_bytes = (
        b"agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        b"102,2000.00,0.10,500,500,1700000000000,false\n"
        b"101,1999.00,0.20,501,501,1700000001000,true\n"
    )
    with ZipFile(source, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("trades.csv", csv_bytes)
    digest = sha256(source.read_bytes()).hexdigest()
    with pytest.raises(binance_archive.DataError, match="identity, order"):
        binance_archive.archive_events(
            source, stream="aggTrades", instrument="ETHUSDT-PERP.SIM",
            source_url="fixture:aggTrades", expected_sha256=digest, delay_ns=1_000_000,
        )


def test_monthly_archive_import_uses_three_checksums_and_clips_inclusive_days(tmp_path, monkeypatch):
    fetched = []

    def fake_fetch(relative_url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")
        fetched.append(relative_url)
        return {"path": str(destination), "url": "fixture:" + relative_url, "sha256": "b" * 64}

    def timestamp(day, hour=0, minute=1):
        epoch_ordinal = date(1970, 1, 1).toordinal()
        return ((date.fromisoformat(day).toordinal() - epoch_ordinal) * DAY_NS
                + hour * 60 * MINUTE_NS + minute * MINUTE_NS)

    def fake_archive_events(path, *, stream, instrument, source_url, expected_sha256, delay_ns):
        month = path.stem[-7:]
        assert month == "2024-01"
        if stream == "fundingRate":
            settlements = (timestamp("2024-01-01", minute=-1), timestamp("2024-01-31", minute=-1),
                           timestamp("2024-02-01", minute=-1))
            result = []
            for index, ts in enumerate(settlements):
                for kind, payload in (
                    ("funding_settlement", {"rate": "0.0001", "settlement_ns": ts,
                                             "interval_minutes": 480}),
                    ("funding_update", {"rate": "0.0001", "rate_semantics": "last_settled_rate"}),
                ):
                    result.append(Event(f"monthly-{kind}-{index}", kind, ts, ts + delay_ns,
                                        ts + delay_ns, Availability.DERIVED, expected_sha256,
                                        {"instrument": instrument, **payload}, source_url))
            return tuple(result)

        kind = {"klines": "ohlcv_1m", "markPriceKlines": "mark",
                "indexPriceKlines": "index"}[stream]
        rows = (timestamp("2024-01-01"), timestamp("2024-02-01", minute=0),
                timestamp("2024-02-01"))
        payload = ({"open": "100", "high": "101", "low": "99", "close": "100", "volume": "2"}
                   if kind == "ohlcv_1m" else {"price": "100"})
        return tuple(Event(f"monthly-{kind}-{index}", kind, ts, ts + delay_ns, ts + delay_ns,
                           Availability.DERIVED, expected_sha256,
                           {"instrument": instrument, **payload}, source_url)
                     for index, ts in enumerate(rows))

    monkeypatch.setattr(binance_archive, "fetch_archive", fake_fetch)
    monkeypatch.setattr(binance_archive, "archive_events", fake_archive_events)
    output = tmp_path / "bundle"
    (output / "raw").mkdir(parents=True)
    result = binance_archive.import_range(
        symbol="ETHUSDT", start_day="2024-01-31", end_day="2024-01-31",
        output=output, delay_ns=1_000_000, archive_interval="monthly",
    )

    assert len(fetched) == 4
    assert all(url.startswith("monthly/") for url in fetched)
    store = read_curated(output / "market.parquet")
    assert sum(event.kind == "ohlcv_1m" for event in store.events) == 1
    assert sum(event.kind == "mark" for event in store.events) == 1
    assert sum(event.kind == "index" for event in store.events) == 1
    assert sum(event.kind == "funding_settlement" for event in store.events) == 1
    assert result["archive_interval"] == "monthly"


def test_funding_api_events_preserve_settlement_facts_and_observed_interval():
    import json

    rows = [
        {"symbol": "ETHUSDT", "fundingTime": 1_800_000_000_000,
         "fundingRate": "0.0001", "markPrice": "2000", "rateType": "Regular"},
        {"symbol": "ETHUSDT", "fundingTime": 1_800_028_800_000,
         "fundingRate": "-0.0002", "markPrice": "2010", "rateType": "Regular"},
    ]
    content = json.dumps(rows).encode("utf-8")
    digest = sha256(content).hexdigest()

    events = binance_archive.funding_api_events(
        content, symbol="ETHUSDT", instrument="ETHUSDT-PERP.SIM",
        source_url="https://fapi.binance.com/fapi/v1/fundingRate?fixture",
        expected_sha256=digest, delay_ns=1_000_000,
    )

    settlements = [event for event in events if event.kind == "funding_settlement"]
    updates = [event for event in events if event.kind == "funding_update"]
    assert [event.payload["settlement_ns"] for event in settlements] == [
        row["fundingTime"] * 1_000_000 for row in rows
    ]
    assert all(event.payload["interval_minutes"] == 480 for event in settlements)
    assert all(event.payload["rate_semantics"] == "last_settled_rate" for event in updates)
    assert all(event.availability_source == Availability.DERIVED for event in events)
    assert all(event.available_at_ns == event.ts_event_ns + 1_000_000 for event in events)
    assert all(event.source_hash == digest for event in events)


def test_funding_api_events_reject_unsorted_or_non_minute_settlement_gaps():
    import json

    base = {"symbol": "ETHUSDT", "fundingRate": "0.0001", "markPrice": "2000",
            "rateType": "Regular"}
    for timestamps in ((1_800_028_800_000, 1_800_000_000_000),
                       (1_800_000_000_000, 1_800_000_030_000)):
        content = json.dumps([dict(base, fundingTime=ts) for ts in timestamps]).encode("utf-8")
        with pytest.raises(ValueError):
            binance_archive.funding_api_events(
                content, symbol="ETHUSDT", instrument="ETHUSDT-PERP.SIM", source_url="fixture",
                expected_sha256=sha256(content).hexdigest(), delay_ns=1_000_000,
            )


def test_import_range_uses_rest_only_for_missing_current_month_funding(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone
    from urllib.error import HTTPError

    today = datetime.now(timezone.utc).date()
    month_start = today.replace(day=1)
    day = month_start.isoformat()
    base_ns = ((month_start - date(1970, 1, 1)).days * DAY_NS)
    rows = [
        {"symbol": "ETHUSDT", "fundingTime": base_ns // 1_000_000,
         "fundingRate": "0.0001", "markPrice": "2000", "rateType": "Regular"},
        {"symbol": "ETHUSDT", "fundingTime": base_ns // 1_000_000 + 28_800_000,
         "fundingRate": "0.0002", "markPrice": "2001", "rateType": "Regular"},
    ]
    content = json.dumps(rows).encode("utf-8")
    digest = sha256(content).hexdigest()

    def fake_fetch(relative_url, destination):
        if relative_url.startswith("monthly/"):
            raise HTTPError("https://data.binance.vision/" + relative_url, 404, "Not Found", None, None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")
        return {"path": str(destination), "url": "fixture:" + relative_url, "sha256": "a" * 64}

    def fake_archive_events(path, *, stream, instrument, source_url, expected_sha256, delay_ns):
        ts = base_ns + DAY_NS - MINUTE_NS
        kind = {"klines": "ohlcv_1m", "markPriceKlines": "mark", "indexPriceKlines": "index"}[stream]
        payload = ({"open": "100", "high": "101", "low": "99", "close": "100", "volume": "2"}
                   if kind == "ohlcv_1m" else {"price": "100"})
        return (Event(f"{stream}-{day}", kind, ts, ts + delay_ns, ts + delay_ns,
                      Availability.DERIVED, expected_sha256, {"instrument": instrument, **payload}, source_url),)

    monkeypatch.setattr(binance_archive, "fetch_archive", fake_fetch)
    monkeypatch.setattr(binance_archive, "archive_events", fake_archive_events)
    monkeypatch.setattr(binance_archive, "fetch_funding_api", lambda *_: (
        {"path": "fixture.json", "url": "https://fapi.binance.com/fapi/v1/fundingRate?fixture",
         "sha256": digest, "verification": "captured_response_sha256"}, content,
    ))
    output = tmp_path / "bundle"

    result = binance_archive.import_range(symbol="ETHUSDT", start_day=day, end_day=day,
                                          output=output, delay_ns=1_000_000)

    assert result["sources"][-1]["verification"] == "captured_response_sha256"
    store = read_curated(output / "market.parquet")
    assert sum(event.kind == "funding_settlement" for event in store.events) == 2


def test_import_range_does_not_use_rest_to_hide_missing_past_archive(tmp_path, monkeypatch):
    from urllib.error import HTTPError

    def fake_fetch(relative_url, destination):
        if relative_url.startswith("monthly/"):
            raise HTTPError("https://data.binance.vision/" + relative_url, 404, "Not Found", None, None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")
        return {"path": str(destination), "url": "fixture:" + relative_url, "sha256": "a" * 64}

    monkeypatch.setattr(binance_archive, "fetch_archive", fake_fetch)
    monkeypatch.setattr(binance_archive, "archive_events", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(binance_archive, "fetch_funding_api", lambda *_: pytest.fail("past-month REST fallback"))
    with pytest.raises(HTTPError):
        binance_archive.import_range(symbol="ETHUSDT", start_day="2024-02-01", end_day="2024-02-01",
                                     output=tmp_path / "bundle", delay_ns=1_000_000)


def test_fetch_archive_retries_transient_https_failures_then_checks_official_hash(tmp_path, monkeypatch):
    import io
    from urllib.error import URLError

    content = b"official archive bytes"
    checksum = sha256(content).hexdigest()
    relative_url = "daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-04-15.zip"
    responses = {
        binance_archive.BASE + relative_url + ".CHECKSUM":
            f"{checksum} ETHUSDT-1m-2024-04-15.zip\n".encode("ascii"),
        binance_archive.BASE + relative_url: content,
    }
    attempts = {url: 0 for url in responses}

    def flaky_urlopen(url, *, timeout):
        assert timeout == 20
        attempts[url] += 1
        if attempts[url] < 3:
            raise URLError("transient TLS EOF")
        return io.BytesIO(responses[url])

    monkeypatch.setattr(binance_archive, "urlopen", flaky_urlopen)
    monkeypatch.setattr(binance_archive.time, "sleep", lambda _delay: None)
    destination = tmp_path / "ETHUSDT-1m-2024-04-15.zip"

    result = binance_archive.fetch_archive(relative_url, destination)

    assert attempts == {url: 3 for url in responses}
    assert destination.read_bytes() == content
    assert result["sha256"] == checksum


def test_fetch_archive_survives_four_consecutive_tls_eofs(tmp_path, monkeypatch):
    import io
    from urllib.error import URLError

    content = b"official archive after transient disconnects"
    checksum = sha256(content).hexdigest()
    relative_url = "daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-04-15.zip"
    responses = {
        binance_archive.BASE + relative_url + ".CHECKSUM":
            f"{checksum} ETHUSDT-1m-2024-04-15.zip\n".encode("ascii"),
        binance_archive.BASE + relative_url: content,
    }
    attempts = {url: 0 for url in responses}

    def flaky_urlopen(url, *, timeout):
        assert timeout == 20
        attempts[url] += 1
        if attempts[url] < 5:
            raise URLError("transient TLS EOF")
        return io.BytesIO(responses[url])

    monkeypatch.setattr(binance_archive, "urlopen", flaky_urlopen)
    monkeypatch.setattr(binance_archive.time, "sleep", lambda _delay: None)
    destination = tmp_path / "ETHUSDT-1m-2024-04-15.zip"

    result = binance_archive.fetch_archive(relative_url, destination)

    assert attempts == {url: 5 for url in responses}
    assert destination.read_bytes() == content
    assert result["sha256"] == checksum


def test_fetch_archive_resume_uses_local_verified_receipt_and_rehashes_bytes(tmp_path, monkeypatch):
    import io

    content = b"official checksum already accepted in a prior interrupted import"
    checksum = sha256(content).hexdigest()
    relative_url = "daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-04-15.zip"
    responses = {
        binance_archive.BASE + relative_url + ".CHECKSUM":
            f"{checksum} ETHUSDT-1m-2024-04-15.zip\n".encode("ascii"),
        binance_archive.BASE + relative_url: content,
    }
    calls = []

    def urlopen(url, *, timeout):
        assert timeout == 20
        calls.append(url)
        return io.BytesIO(responses[url])

    monkeypatch.setattr(binance_archive, "urlopen", urlopen)
    destination = tmp_path / "ETHUSDT-1m-2024-04-15.zip"
    first = binance_archive.fetch_archive(relative_url, destination)
    assert destination.with_suffix(".zip.source.json").is_file()

    monkeypatch.setattr(binance_archive, "urlopen", lambda *_args, **_kwargs: pytest.fail("receipt should avoid network"))
    second = binance_archive.fetch_archive(relative_url, destination)
    assert second == first
    assert calls == [binance_archive.BASE + relative_url + ".CHECKSUM",
                     binance_archive.BASE + relative_url]

    destination.write_bytes(b"tampered cached bytes")
    with pytest.raises(binance_archive.DataError, match="verified source receipt"):
        binance_archive.fetch_archive(relative_url, destination)


def test_fetch_archive_does_not_write_content_with_wrong_official_hash(tmp_path, monkeypatch):
    import io

    content = b"downloaded but not matching"
    relative_url = "daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-04-15.zip"

    def fake_urlopen(url, *, timeout):
        assert timeout == 20
        if url.endswith(".CHECKSUM"):
            return io.BytesIO(("0" * 64 + " ETHUSDT-1m-2024-04-15.zip\n").encode("ascii"))
        return io.BytesIO(content)

    monkeypatch.setattr(binance_archive, "urlopen", fake_urlopen)
    destination = tmp_path / "ETHUSDT-1m-2024-04-15.zip"

    with pytest.raises(binance_archive.DataError, match="checksum mismatch"):
        binance_archive.fetch_archive(relative_url, destination)

    assert not destination.exists()


def test_fetch_archive_uses_verified_tls_curl_fallback_after_python_retries(tmp_path, monkeypatch):
    import subprocess
    from urllib.error import URLError

    content = b"archive bytes fetched by the platform TLS client"
    checksum = sha256(content).hexdigest()
    relative_url = "daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-04-15.zip"
    responses = {
        binance_archive.BASE + relative_url + ".CHECKSUM":
            f"{checksum} ETHUSDT-1m-2024-04-15.zip\n".encode("ascii"),
        binance_archive.BASE + relative_url: content,
    }
    curl_calls = []

    def broken_urlopen(_url, *, timeout):
        assert timeout == 20
        raise URLError("transient TLS EOF")

    def fake_run(command, *, capture_output, check, timeout):
        assert capture_output and check is False and timeout == 25
        assert "--retry-all-errors" in command
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(responses[command[-1]])
        curl_calls.append(command[-1])
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(binance_archive, "urlopen", broken_urlopen)
    monkeypatch.setattr(binance_archive.shutil, "which", lambda _name: "curl.exe")
    monkeypatch.setattr(binance_archive.subprocess, "run", fake_run)
    monkeypatch.setattr(binance_archive.time, "sleep", lambda _delay: None)
    destination = tmp_path / "ETHUSDT-1m-2024-04-15.zip"

    result = binance_archive.fetch_archive(relative_url, destination)

    assert curl_calls == list(responses)
    assert destination.read_bytes() == content
    assert result["sha256"] == checksum


@pytest.mark.parametrize(
    ("start_day", "end_day", "delay_ns"),
    [("2024-02-30", "2024-03-01", 1), ("2024-03-02", "2024-03-01", 1),
     ("2024-03-01", "2024-03-01", 0)],
)
def test_import_range_rejects_invalid_request_before_fetch(
    tmp_path, monkeypatch, start_day, end_day, delay_ns,
):
    monkeypatch.setattr(binance_archive, "fetch_archive", lambda *_: pytest.fail("must not fetch"))
    with pytest.raises(ValueError):
        binance_archive.import_range(symbol="ETHUSDT", start_day=start_day, end_day=end_day,
                                     output=tmp_path / "bundle", delay_ns=delay_ns)


def test_bookticker_import_is_bounded_to_one_day(tmp_path, monkeypatch):
    monkeypatch.setattr(binance_archive, "fetch_archive", lambda *_: pytest.fail("must not fetch"))
    with pytest.raises(ValueError, match="limited to one UTC day"):
        binance_archive.import_range(symbol="ETHUSDT", start_day="2024-01-01",
                                     end_day="2024-01-02", output=tmp_path / "bundle",
                                     delay_ns=1_000_000, include_bookticker=True)


def test_one_day_range_import_can_include_bookticker(tmp_path, monkeypatch):
    fetched = []

    def fake_fetch(relative_url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")
        fetched.append(relative_url)
        return {"path": str(destination), "url": "fixture:" + relative_url, "sha256": "a" * 64}

    def fake_archive_events(path, *, stream, instrument, source_url, expected_sha256, delay_ns):
        ts = 1_704_067_260_000_000_000
        if stream == "fundingRate":
            facts = (("funding_settlement", {"rate": "0.0001", "settlement_ns": ts,
                                              "interval_minutes": 480}),
                     ("funding_update", {"rate": "0.0001", "rate_semantics": "last_settled_rate"}))
        elif stream == "klines":
            facts = (("ohlcv_1m", {"open": "100", "high": "101", "low": "99",
                                   "close": "100", "volume": "2"}),)
        elif stream in {"markPriceKlines", "indexPriceKlines"}:
            facts = (("mark" if stream == "markPriceKlines" else "index", {"price": "100"}),)
        else:
            facts = (("best_quote", {"bid_price": "100", "bid_qty": "0.1",
                                      "ask_price": "100.01", "ask_qty": "0.2",
                                      "update_id": 1, "transaction_time_ns": ts}),)
        return tuple(Event(f"{stream}-{kind}", kind, ts, ts + delay_ns, ts + delay_ns,
                           Availability.DERIVED, expected_sha256,
                           {"instrument": instrument, **payload}, source_url)
                     for kind, payload in facts)

    monkeypatch.setattr(binance_archive, "fetch_archive", fake_fetch)
    monkeypatch.setattr(binance_archive, "archive_events", fake_archive_events)
    result = binance_archive.import_range(symbol="ETHUSDT", start_day="2024-01-01",
                                          end_day="2024-01-01", output=tmp_path / "bundle",
                                          delay_ns=1_000_000, include_bookticker=True)

    store = read_curated(tmp_path / "bundle" / "market.parquet")
    assert sum(event.kind == "best_quote" for event in store.events) == 1
    assert any("daily/bookTicker/" in url for url in fetched)
    assert result["bookticker_sampling"] == "last_event_time_quote_per_second"


def test_bookticker_archive_keeps_last_real_quote_per_event_second(tmp_path):
    import csv
    import io

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("update_id", "best_bid_price", "best_bid_qty", "best_ask_price",
                     "best_ask_qty", "transaction_time", "event_time"))
    writer.writerow((100, "2000.00", "0.1", "2000.10", "0.2", 1_704_067_200_000, 1_704_067_200_010))
    writer.writerow((101, "2000.05", "0.3", "2000.15", "0.4", 1_704_067_200_500, 1_704_067_200_510))
    writer.writerow((102, "2000.10", "0.5", "2000.20", "0.6", 1_704_067_201_000, 1_704_067_201_010))
    contents = BytesIO()
    with ZipFile(contents, "w", ZIP_DEFLATED) as archive:
        archive.writestr("ETHUSDT-bookTicker-2024-01-01.csv", output.getvalue())
    zipped = contents.getvalue()
    path = tmp_path / "bookticker.zip"
    path.write_bytes(zipped)
    checksum = sha256(zipped).hexdigest()

    events = binance_archive.archive_events(
        path, stream="bookTicker", instrument="ETHUSDT-PERP.SIM",
        source_url="fixture://bookticker", expected_sha256=checksum, delay_ns=1_000_000,
    )

    assert len(events) == 2
    assert events[0].kind == "best_quote"
    assert events[0].payload["update_id"] == 101
    assert events[0].payload["bid_price"] == "2000.05"
    assert events[0].payload["ask_qty"] == "0.4"
    assert events[0].ts_init_ns == events[0].ts_event_ns + 1_000_000
    assert events[0].provenance.endswith("#last-event-time-quote-per-second")
