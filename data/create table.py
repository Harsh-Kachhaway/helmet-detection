import sqlite3

def create_and_insert_sample_plate_data():
    # Sample data for license plates, owner names, and phone numbers
    sample_data = [
        ("AB1234", "John Doe", "+1234567890"),
        ("CD5678", "Jane Smith", "+0987654321"),
        ("EF9101", "Alice Brown", "+1122334455"),
        ("GH1122", "Bob White", "+9988776655"),
        ("IJ3344", "Charlie Green", "+2233445566")
    ]

    # Connect to SQLite database (creates the file if it doesn't exist)
    conn = sqlite3.connect('detection_data.db')
    cursor = conn.cursor()

    # Create the 'plates' table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS plates (
        plate_text TEXT PRIMARY KEY,
        owner_name TEXT,
        phone_number TEXT
    )
    """)

    # Insert sample data into 'plates' table
    cursor.executemany("""
    INSERT OR IGNORE INTO plates (plate_text, owner_name, phone_number)
    VALUES (?, ?, ?)
    """, sample_data)

    # Commit and close the database connection
    conn.commit()
    conn.close()

# Call the function to create the table and insert data
create_and_insert_sample_plate_data()
