from datetime import datetime

from jwtauth.database import Base
from sqlalchemy import Column, Integer, String, SmallInteger, DateTime


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(32), nullable=False, unique=True)
    password = Column(String(264), nullable=False)
    role = Column(SmallInteger, nullable=True, default=0)
    updated = Column(DateTime, nullable=True, default=datetime.utcnow)

    def __init__(self, email, password, role = 0):
        self.email = email
        self.password = password
        self.role = role

    def __repr__(self):
        return f'<User {{ email: {self.email}, \
password: {self.password} }}'