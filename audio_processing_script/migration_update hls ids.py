import hashlib
from sqlalchemy import create_engine, MetaData, Table, select, update

# Replace with your actual DB URL
DATABASE_URL = "postgresql://user:password@localhost:5432/yourdb"

# Setup
engine = create_engine(DATABASE_URL)
connection = engine.connect()
metadata = MetaData(bind=engine)

# Reflect the existing table
tracks_table = Table('tracks', metadata, autoload_with=engine)

def generate_track_id(s3_key: str) -> str:
    short_hash = hashlib.md5(s3_key.encode()).hexdigest()[:10]
    return f"track_{short_hash}"

# Step 1: Fetch all rows missing track_id
select_query = select(tracks_table).where(tracks_table.c.track_id == None)
results = connection.execute(select_query).fetchall()

updated = 0

# Step 2: Process each row
for row in results:
    original_s3_key = row.original_s3_key
    if not original_s3_key:
        continue

    track_id = generate_track_id(original_s3_key)

    # Step 3: Update the record
    update_query = (
        update(tracks_table)
        .where(tracks_table.c.id == row.id)
        .values(track_id=track_id)
    )
    connection.execute(update_query)
    updated += 1

print(f"✅ Updated {updated} records with new track IDs.")

# Cleanup
connection.close()
