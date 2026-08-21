import sqlite3
import pandas as pd

DB_NAME = "real_estate.db"


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    return sqlite3.connect(DB_NAME)


# =========================================================
# CREATE DATABASE
# =========================================================

def create_database():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            bhk INTEGER NOT NULL,
            area INTEGER NOT NULL,
            property_type TEXT NOT NULL,
            price REAL NOT NULL,
            status TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            interested_property TEXT,
            budget REAL,
            status TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# =========================================================
# PROPERTY MANAGEMENT
# =========================================================

def add_property(
    location,
    bhk,
    area,
    property_type,
    price,
    status
):

    conn = get_connection()

    conn.execute("""
        INSERT INTO properties
        (
            location,
            bhk,
            area,
            property_type,
            price,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        location,
        bhk,
        area,
        property_type,
        price,
        status
    ))

    conn.commit()
    conn.close()


def get_properties():

    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM properties",
        conn
    )

    conn.close()

    df = df.rename(columns={
        "id": "ID",
        "location": "Location",
        "bhk": "BHK",
        "area": "Area",
        "property_type": "Property_Type",
        "price": "Price",
        "status": "Status"
    })

    return df


def update_property(
    property_id,
    location,
    bhk,
    area,
    property_type,
    price,
    status
):

    conn = get_connection()

    conn.execute("""
        UPDATE properties
        SET
            location = ?,
            bhk = ?,
            area = ?,
            property_type = ?,
            price = ?,
            status = ?
        WHERE id = ?
    """, (
        location,
        bhk,
        area,
        property_type,
        price,
        status,
        property_id
    ))

    conn.commit()
    conn.close()


def delete_property(property_id):

    conn = get_connection()

    conn.execute("""
        DELETE FROM properties
        WHERE id = ?
    """, (
        property_id,
    ))

    conn.commit()
    conn.close()


# =========================================================
# LEAD MANAGEMENT
# =========================================================

def add_lead(
    customer_name,
    phone,
    interested_property,
    budget,
    status
):

    conn = get_connection()

    conn.execute("""
        INSERT INTO leads
        (
            customer_name,
            phone,
            interested_property,
            budget,
            status
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        customer_name,
        phone,
        interested_property,
        budget,
        status
    ))

    conn.commit()
    conn.close()


def get_leads():

    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM leads",
        conn
    )

    conn.close()

    df = df.rename(columns={
        "id": "ID",
        "customer_name": "Customer Name",
        "phone": "Phone",
        "interested_property": "Interested Property",
        "budget": "Budget (₹ Lakh)",
        "status": "Status"
    })

    return df


def update_lead_status(lead_id, new_status):

    conn = get_connection()

    conn.execute("""
        UPDATE leads
        SET status = ?
        WHERE id = ?
    """, (
        new_status,
        lead_id
    ))

    conn.commit()
    conn.close()