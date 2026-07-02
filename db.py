import sqlite3

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
        data_train BLOB,
        data_val BLOB,
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

def get_raw_bytes(filename):
    return open(filename,"rb").read()

def save_raw_bytes(data,filename):
    fd = open(filename,"wb")
    fd.write(data)
    fd.close()

def add_entry_lang(name,size_sigma,nb_words,len_words,size_auto,reset,data):
    global DB
    curr = get_cursor()
    curr.execute("INSERT INTO Languages (name,size_sigma,nb_words,len_words,size_auto,reset,data) VALUES (?,?,?,?,?,?,?);",(name,size_sigma,nb_words,len_words,size_auto,reset,data))
    DB.commit()

def add_entry_dataset(lang,name,label,nb_train,len_train,nb_val,len_val,nb_test,len_test,data_train,data_val,data_test):
    global DB
    curr = get_cursor()
    curr.execute("INSERT INTO Datasets (lang,name,label,nb_train,len_train,nb_val,len_val,nb_test,len_test,data_train,data_val,data_test ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);",(lang,name,label,nb_train,len_train,nb_val,len_val,nb_test,len_test,data_train,data_val,data_test))
    DB.commit()
    pass

def add_entry_models(
        lang ,
        dataset ,
        name ,
        epochs ,
        weights ,
        data):
    global DB
    curr = get_cursor()
    curr.execute("INSERT INTO Models (lang,dataset,name,epochs,weights,data) VALUES (?,?,?,?,?,?);",(lang,dataset,name,epochs,weights,data))
    DB.commit()

def add_entry_auto(
    lang ,
    dataset ,
    model ,
    name ,
    nb_explore ,
    nb_clusters ,
    nb_etats ,
    data ):
    global DB
    curr = get_cursor()
    curr.execute("INSERT INTO Models (lang ,dataset ,model ,name ,nb_explore ,nb_clusters ,nb_etats ,data ) VALUES (?,?,?,?,?,?,?,?);",(lang ,dataset ,model ,name ,nb_explore ,nb_clusters ,nb_etats ,data))
    DB.commit()


def get_data(command):
    global DB
    curr = get_cursor()
    curr.execute(command)
    return curr.fetchall()

if __name__ == "__main__":
    setup_db()
    print(get_data("SELECT * FROM Models;"))
