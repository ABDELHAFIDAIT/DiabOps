from pydantic import BaseModel, Field


class PatientData(BaseModel):
    Pregnancies : int                   = Field(..., example=2)
    Glucose : float                     = Field(..., example=90)
    BloodPressure : float               = Field(..., example=80)
    SkinThickness : float               = Field(..., example=30)
    Insulin : float                     = Field(..., example=5)
    BMI : float                         = Field(..., example=20.6)
    DiabetesPedigreeFunction : float    = Field(..., example=0.627)
    Age : int                           = Field(..., example=30)


class PredictionResponse(BaseModel):
    prediction : int
    risk_level : str
    probability : float | None
    