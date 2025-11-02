from typing import Dict, List, Optional, Union
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Запрос инференса: один объект с набором фичей.

    Все поля динамические. Представляем как словарь: имя фичи -> значение.
    Поддерживаются числовые и строковые значения, а также None.
    """

    features: Dict[str, Optional[Union[float, int, str]]] = Field(
        ...,
        description="Словарь признаков: имя -> значение (float|int|string|None)",
    )


class PredictResponse(BaseModel):
    """Ответ инференса: предсказания для объекта с метаданными запроса."""

    request_id: UUID = Field(
        ...,
        description="Уникальный идентификатор запроса (UUID)"
    )
    timestamp: str = Field(
        ...,
        description="Время и дата ответа сервиса в формате ISO 8601"
    )
    predictions: List[float] = Field(
        ...,
        description="Список предсказаний (для мультитаргетной модели может быть несколько значений)"
    )
