from alembic import context
from app.infrastructure.config import get_settings
from app.infrastructure.database import Base, create_session_factory

config = context.config
target_metadata = Base.metadata
settings = get_settings()
if settings.database_url is None:
    raise ValueError("DATABASE_URL is required for migrations")

if context.is_offline_mode():
    context.configure(url=settings.database_url.get_secret_value(),
                      target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()
else:
    engine, _ = create_session_factory(settings)
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()
