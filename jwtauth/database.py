from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

DATABASE_URL = 'mysql+pymysql://root:@localhost/atcv'

engine = create_engine(DATABASE_URL, convert_unicode=True)

db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()


def init_db():
    Base.metadata.create_all(bind=engine)


def shutdown_db_session():
    db_session.remove()