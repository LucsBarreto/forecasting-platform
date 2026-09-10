"""
engenharia de features temporais.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import settings
from src.config.pipeline import TemporalFeatureSettings
from src.core.exceptions.validation import DataValidationError


HOLIDAYS_FILE = Path("configs/holidays.yaml")


@dataclass(slots=True, frozen=True)
class HolidayCalendar:
    """calendário de feriados normalizado para datas e nomes."""

    dates: set[pd.Timestamp]

    names: dict[pd.Timestamp, str]


@dataclass(slots=True)
class TemporalFeatureEngineer:
    """cria features temporais a partir da coluna de data."""

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        executa a engenharia de features temporais.

        parameters
        ----------
        dataframe
            dataframe de entrada.

        returns
        -------
        pd.dataframe
            dataframe enriquecido com features temporais.
        """

        data = dataframe.copy()

        config: TemporalFeatureSettings = (
            settings
            .pipeline
            .preprocessing
            .feature_engineering
            .temporal_features
        )

        date_column = config.date_column

        if date_column not in data.columns:
            raise DataValidationError(
                f"Date column '{date_column}' was not found."
            )

        if not pd.api.types.is_datetime64_any_dtype(
            data[date_column],
        ):
            raise DataValidationError(
                f"Date column '{date_column}' must be datetime."
            )

        holidays_calendar = self._load_holidays(
            HOLIDAYS_FILE,
            data,
            date_column,
        )

        holiday_dates = holidays_calendar.dates

        data = self._create_calendar_features(
            data=data,
            date_column=date_column,
            holiday_dates=holiday_dates,
            config=config,
        )

        data = self._create_distance_features(
            data=data,
            date_column=date_column,
            holiday_dates=holiday_dates,
            config=config,
        )

        data = self._create_cyclical_features(
            data=data,
            date_column=date_column,
            config=config,
        )

        return data

    @staticmethod
    def _load_holidays(
        path: Path,
        data: pd.DataFrame | None = None,
        date_column: str | None = None,
    ) -> HolidayCalendar:
        """carrega o calendário em yaml com fallback mínimo para json."""

        file_path = path if path.exists() else Path(str(path).replace(".yaml", ".json"))

        if not file_path.exists():
            raise FileNotFoundError(f"arquivo de feriados não encontrado: {file_path}")

        with file_path.open(mode="r", encoding="utf-8") as file:
            if file_path.suffix == ".yaml":
                payload = yaml.safe_load(file)
            else:
                import json
                payload = json.load(file)

        holiday_rules = TemporalFeatureEngineer._flatten_holiday_rules(payload)

        min_year = 2025
        max_year = 2025
        if data is not None and date_column and date_column in data.columns:
            dates = pd.to_datetime(data[date_column], errors="coerce")
            dates = dates.dropna()
            if not dates.empty:
                min_year = int(dates.dt.year.min())
                max_year = int(dates.dt.year.max())

        holiday_dates: set[pd.Timestamp] = set()
        holiday_names: dict[pd.Timestamp, str] = {}

        for rule in holiday_rules:
            if not isinstance(rule, dict):
                continue

            name = str(rule.get("name", "feriado"))

            if "date" in rule:
                parsed_date = pd.Timestamp(rule["date"]).normalize()
                holiday_dates.add(parsed_date)
                holiday_names[parsed_date] = name
                continue

            if rule.get("base") == "easter":
                offset = int(rule.get("offset_days", 0))
                for year in range(min_year, max_year + 1):
                    easter_date = TemporalFeatureEngineer._easter_sunday(year)
                    target = easter_date + timedelta(days=offset)
                    normalized = pd.Timestamp(target).normalize()
                    holiday_dates.add(normalized)
                    holiday_names[normalized] = name
                continue

            if rule.get("rule"):
                for year in range(min_year, max_year + 1):
                    date_value = TemporalFeatureEngineer._resolve_rule_date(year, rule["rule"])
                    if date_value is not None:
                        normalized = pd.Timestamp(date_value).normalize()
                        holiday_dates.add(normalized)
                        holiday_names[normalized] = name
                continue

            if "month" in rule and "day" in rule:
                for year in range(min_year, max_year + 1):
                    try:
                        parsed_date = pd.Timestamp(year, int(rule["month"]), int(rule["day"])).normalize()
                    except Exception:
                        continue
                    holiday_dates.add(parsed_date)
                    holiday_names[parsed_date] = name

        return HolidayCalendar(dates=holiday_dates, names=holiday_names)

    @staticmethod
    def _flatten_holiday_rules(payload):
        """normaliza o contrato yaml de feriados em regras."""

        if isinstance(payload, list):
            return payload

        if not isinstance(payload, dict):
            return []

        if "holidays" in payload:
            return list(payload.get("holidays", []))

        rules: list[dict] = []

        for section in ("national", "commercial", "observance"):
            items = payload.get(section, [])
            if isinstance(items, list):
                rules.extend(items)

        regional = payload.get("regional", {})
        if isinstance(regional, dict):
            rules.extend(regional.get("state", []))
            rules.extend(regional.get("municipal", []))

        return rules

    @staticmethod
    def _resolve_rule_date(year: int, rule: str) -> pd.Timestamp | None:
        """resolve regras específicas do calendário para o ano informado."""

        if rule == "second_sunday_of_may":
            return TemporalFeatureEngineer._nth_sunday_of_month(year, 5, 2)
        if rule == "second_sunday_of_august":
            return TemporalFeatureEngineer._nth_sunday_of_month(year, 8, 2)
        if rule == "last_friday_of_november":
            return TemporalFeatureEngineer._last_weekday_of_month(year, 11, "FRIDAY")
        if rule == "monday_after_black_friday":
            black_friday = TemporalFeatureEngineer._last_weekday_of_month(year, 11, "FRIDAY")
            if black_friday is None:
                return None
            return black_friday + timedelta(days=3)
        if rule == "winter_start":
            return pd.Timestamp(year, 6, 21)
        if rule == "spring_start":
            return pd.Timestamp(year, 9, 23)
        return None

    @staticmethod
    def _nth_sunday_of_month(year: int, month: int, occurrence: int) -> pd.Timestamp:
        """retorna o domingo da ocorrência indicada no mês."""

        first_day = pd.Timestamp(year, month, 1)
        first_sunday = first_day + timedelta(days=(6 - first_day.dayofweek) % 7)
        return first_sunday + timedelta(days=7 * (occurrence - 1))

    @staticmethod
    def _last_weekday_of_month(year: int, month: int, weekday: str) -> pd.Timestamp | None:
        """retorna o último dia da semana informado no mês."""

        last_day = pd.Timestamp(year, month + 1, 1) - timedelta(days=1)
        target = last_day.dayofweek
        weekday_map = {
            "MONDAY": 0,
            "TUESDAY": 1,
            "WEDNESDAY": 2,
            "THURSDAY": 3,
            "FRIDAY": 4,
            "SATURDAY": 5,
            "SUNDAY": 6,
        }

        delta = (target - weekday_map[weekday]) % 7
        return last_day - timedelta(days=delta)

    @staticmethod
    def _easter_sunday(year: int) -> pd.Timestamp:
        """retorna o domingo de páscoa para o ano informado."""

        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return pd.Timestamp(year, month, day)

    @staticmethod
    def _create_calendar_features(
        data: pd.DataFrame,
        date_column: str,
        holiday_dates: set[pd.Timestamp],
        config: TemporalFeatureSettings,
    ) -> pd.DataFrame:
        """cria features baseadas no calendário."""

        dates = data[date_column].dt.normalize()

        is_holiday = dates.isin(
            holiday_dates,
        )

        if config.create_business_day:

            data["is_business_day"] = (
                (dates.dt.dayofweek < 5)
                & (~is_holiday)
            )

        if config.create_holiday:

            one_day = timedelta(days=1)

            previous_day = dates - one_day
            next_day = dates + one_day

            data["is_holiday"] = is_holiday

            data["is_holiday_eve"] = (
                next_day.isin(
                    holiday_dates,
                )
            )

            data["is_post_holiday"] = (
                previous_day.isin(
                    holiday_dates,
                )
            )

        return data

    @staticmethod
    def _days_until_holiday(
        current: pd.Timestamp,
        holidays: list[pd.Timestamp],
    ) -> int:
        """
        retorna a quantidade de dias até o próximo feriado.

        returns
        -------
        int
            quantidade de dias até o próximo feriado,
            ou -1 quando não houver feriado futuro.
        """

        for holiday in holidays:

            if holiday >= current:
                return (holiday - current).days

        return -1

    @staticmethod
    def _days_since_holiday(
        current: pd.Timestamp,
        holidays: list[pd.Timestamp],
    ) -> int:
        """
        retorna a quantidade de dias desde o feriado anterior.

        returns
        -------
        int
            quantidade de dias desde o feriado anterior,
            ou -1 quando não houver feriado anterior.
        """

        for holiday in reversed(holidays):

            if holiday <= current:
                return (current - holiday).days

        return -1

    @staticmethod
    def _create_distance_features(
        data: pd.DataFrame,
        date_column: str,
        holiday_dates: set[pd.Timestamp],
        config: TemporalFeatureSettings,
    ) -> pd.DataFrame:
        """cria features temporais baseadas em distância."""

        dates = data[date_column].dt.normalize()

        if config.create_days_to_month_end:

            month_end = dates + pd.offsets.MonthEnd(0)

            data["days_to_month_end"] = (
                month_end - dates
            ).dt.days

            month_start = (
                dates
                .dt
                .to_period("M")
                .dt
                .start_time
            )

            data["days_from_month_start"] = (
                dates - month_start
            ).dt.days

        if config.create_holiday_distance:

            holiday_list = sorted(
                holiday_dates,
            )

            data["days_until_holiday"] = dates.apply(
                lambda current: (
                    TemporalFeatureEngineer
                    ._days_until_holiday(
                        current,
                        holiday_list,
                    )
                )
            )

            data["days_since_holiday"] = dates.apply(
                lambda current: (
                    TemporalFeatureEngineer
                    ._days_since_holiday(
                        current,
                        holiday_list,
                    )
                )
            )

        return data

    @staticmethod
    def _create_cyclical_features(
        data: pd.DataFrame,
        date_column: str,
        config: TemporalFeatureSettings,
    ) -> pd.DataFrame:
        """cria features cíclicas baseadas no calendário."""

        if not config.create_cyclical_features:
            return data

        dates = data[date_column].dt

        month = dates.month
        weekday = dates.dayofweek
        week = dates.isocalendar().week.astype(int)
        day_of_year = dates.dayofyear

        data["month_sin"] = np.sin(
            2 * np.pi * month / 12,
        )

        data["month_cos"] = np.cos(
            2 * np.pi * month / 12,
        )

        data["weekday_sin"] = np.sin(
            2 * np.pi * weekday / 7,
        )

        data["weekday_cos"] = np.cos(
            2 * np.pi * weekday / 7,
        )

        data["week_sin"] = np.sin(
            2 * np.pi * week / 52,
        )

        data["week_cos"] = np.cos(
            2 * np.pi * week / 52,
        )

        data["dayofyear_sin"] = np.sin(
            2 * np.pi * day_of_year / 365,
        )

        data["dayofyear_cos"] = np.cos(
            2 * np.pi * day_of_year / 365,
        )

        return data