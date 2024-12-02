from pydantic import BaseModel, ConfigDict
from datetime import date
from typing import Optional

class EmployeeBase(BaseModel):
    name: str
    position: str
    hire_date: date
    phone_number: str
    emergency_contact: str
    email_address: str
    user_id:int

class EmployeeCreate(EmployeeBase):
    pass

class EmployeeUpdate(EmployeeBase):
    name: Optional[str] = None
    position: Optional[str] = None
    hire_date: Optional[date] = None
    phone_number: Optional[str] = None
    emergency_contact: Optional[str] = None
    email_address: Optional[str] = None
    user_id: Optional[int] = None

class Employee(EmployeeBase):
    id: int
    user_id: int

    # class Config:
    #     orm_mode = True
    model_config = ConfigDict(from_attributes=True)

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    user_id: int
    employees: list[Employee] = []

    model_config = ConfigDict(from_attributes=True)




