from aiogram.dispatcher.filters.state import (StatesGroup, State)


class UserRegistration(StatesGroup):
    first_name = State()
    last_name = State()
    phone_number = State()
    t1 = State()
    t11 = State()
    t2 = State()
    t21 = State()
    t3 = State()
    t31 = State()
    t4 = State()
    t41 = State()
    t5 = State()
    t51 = State()