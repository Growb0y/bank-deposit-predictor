from pydantic import BaseModel, Field
from typing import List, Optional


class ClientData(BaseModel):
    age: int = Field(..., ge=18, le=100, description='Возраст')
    job: str = Field(..., description='Профессия')
    marital: str = Field(..., description='Семейное положение')
    education: str = Field(..., description='Образование')
    default: str = Field(..., description='Дефолт (да/нет)')
    balance: float = Field(..., description='Среднегодовой баланс')
    housing: str = Field(..., description='Жилищный кредит (да/нет)')
    loan: str = Field(..., description='Потребительский кредит (да/нет)')
    contact: str = Field(..., description='Тип контакта')
    month: str = Field(..., description='Месяц последнего контакта')
    duration: float = Field(..., ge=0, description='Длительность звонка (сек)')
    campaign: int = Field(..., ge=1, description='Количество звонков в кампании')
    pdays: int = Field(..., description='Дней после последнего контакта (-1 если не было)')
    previous: int = Field(..., ge=0, description='Контактов до этой кампании')
    poutcome: str = Field(..., description='Результат предыдущей кампании')


class BatchClientData(BaseModel):
    clients: List[ClientData]


class PredictionResult(BaseModel):
    probability: float
    prediction: int
    prediction_label: str


class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]
    summary: dict


class StrategyInput(BaseModel):
    cost_per_call: float = Field(5.0, ge=0, description='Стоимость одного звонка')
    profit_per_deposit: float = Field(50.0, ge=0, description='Прибыль с одного оформленного депозита')
    budget: Optional[float] = Field(None, ge=0, description='Бюджет кампании (опционально)')


class StrategyResult(BaseModel):
    optimal_threshold: float
    total_clients: int
    clients_to_call: int
    expected_deposits: float
    expected_cost: float
    expected_revenue: float
    expected_profit: float
    roi: float
    conversion_rate: float
    budget_used: Optional[float] = None
    recommendations: List[str]
