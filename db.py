import sqlite3
import benchmarksV2

DB = None

def get_cursor():
    global DB
    if DB is None:
        DB = sqlite3.connect("DATABASE.db")
    return DB.cursor()


def setup_db():
    global DB
    curr = get_cursor()
    curr.execute("""CREATE TABLE IF NOT EXISTS Languages(
        id INTEGER  PRIMARY KEY AUTOINCREMENT,
        name STRING,
        size_sigma INTEGER,
        nb_words INTEGER,
        len_words INTEGER,
        size_auto INTEGR,
        reset BOOL,
        data BLOB
    );""")
    curr.execute("""CREATE TABLE IF NOT EXISTS Datasets(
        id INTEGER  PRIMARY KEY AUTOINCREMENT,
        lang INTEGER,
        name STRING,
        label STRING,
        nb_train INTEGER,
        len_train INTEGER,

        nb_val INTEGER,
        len_val INTEGER,

        nb_test INTEGER,
        len_test INTEGER,
        data_train BLOB
        data_val BLOB
        data_test BLOB
    );""")
    curr.execute("""CREATE TABLE IF NOT EXISTS Models(
        id INTEGER  PRIMARY KEY AUTOINCREMENT,
        lang INTEGER,
        dataset INTEGER,
        name STRING,
        epochs INTEGER,
        weights INTEGER,
        data BLOB
    );""")
    curr.execute("""CREATE TABLE IF NOT EXISTS Autos(
        id INTEGER  PRIMARY KEY AUTOINCREMENT,
        lang INTEGER,
        dataset INTEGER,
        model INTEGER,
        name STRING,
        nb_explore INTEGER,
        nb_clusters INTEGER,
        nb_etats INTEGER,
        data BLOB
    );""")

    DB.commit()
    curr.close()

if __name__ == "__main__":
    setup_db()
