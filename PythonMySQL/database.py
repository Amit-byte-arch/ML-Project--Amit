import mysql.connector

conn = mysql.connector.connect(host = 'localhost', user ='root', password='6942')

if conn.is_connected():
    print('connection established')
# print(conn)


mycursor = conn.cursor()
mycursor.execute('create database pythondb')
print(mycursor)

#for x in mycursor:
    #print(x) 