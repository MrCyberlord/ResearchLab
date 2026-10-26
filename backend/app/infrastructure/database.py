from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.infrastructure.config import Settings


class Base(DeclarativeBase):
    pass


def create_session_factory(settings: Settings):
    if settings.database_url is None:
        raise ValueError("DATABASE_URL is required for database operations")
    engine = create_engine(settings.database_url.get_secret_value(), pool_pre_ping=True,
                           connect_args={"connect_timeout": 5})
    return engine, sessionmaker(bind=engine, expire_on_commit=False)
