import pandas as pd
import re

# --- CONFIGURATION ---
# Use the output from the last step (or your original file if you want a fresh start)
INPUT_FILE = "Taxonomy_With_Chemicals.xlsx" 
OUTPUT_FILE = "Taxonomy_Major_Pools.xlsx"
SHEET_NAME = "Combined_Taxonomy"

# --- THE MASTER CLASSIFICATION MAP ---
# (Order matters: We check these from top to bottom)
MASTER_MAP = {
    # --- 1. EQUIPMENT POOL ---
    "Equipment": [
        # Rotating
        "pump", "compressor", "turbine", "fan", "blower", "motor", "engine", "centrifuge", "mixer", "agitator",
        # Static / Vessels
        "tank", "vessel", "reactor", "column", "tower", "boiler", "furnace", "heater", "exchanger", "condenser", "reboiler", "sphere", "drum", "silo", "hopper",
        # Piping & Valves
        "valve", "pipe", "piping", "pipeline", "flange", "gasket", "fitting", "joint", "blind", "spade", "strainer", "filter", "trap", "hose", "loading arm",
        # Instrumentation / Electrical
        "sensor", "transmitter", "gauge", "meter", "analyzer", "detector", "switch", "relay", "cable", "wire", "transformer", "breaker", "battery", "ups", "generator", "instrument", "control loop", "logic", "plc", "dcs", "scada",
        # Safety Systems
        "relief", "psv", "prv", "disk", "vent", "flare", "stack", "scrubber", "absorber", "deluge", "sprinkler", "monitor", "detector", "alarm", "trip", "interlock", "esd", "sis"
    ],

    # --- 2. SUBSTANCES POOL ---
    "Substances": [
        # Chemicals
        "acid", "base", "caustic", "soda", "hydroxide", "ammonia", "chlorine", "sulfur", "nitrogen", "oxygen", "hydrogen", "carbon", "oxide", "peroxide", "catalyst", "polymer", "resin", "powder", "dust",
        # Hydrocarbons / Fuels
        "oil", "gas", "fuel", "diesel", "petrol", "gasoline", "kerosene", "naptha", "crude", "condensate", "lpg", "lng", "methane", "ethane", "propane", "butane", "ethylene", "propylene", "benzene", "toluene", "xylene", "solvent", "hydrocarbon",
        # Utilities / Waste
        "water", "steam", "air", "waste", "effluent", "sludge", "sewage"
    ],

    # --- 3. PROCESS & EVENTS POOL ---
    "Process_Events": [
        # Loss of Containment
        "leak", "spill", "release", "emission", "discharge", "escape", "seepage", "rupture", "burst", "fracture", "crack", "hole", "corrosion", "erosion", "pitting", "fatigue",
        # Fire / Explosion
        "fire", "explosion", "blast", "flash", "ignition", "burn", "flame", "smoke", "fume", "cloud", "vapor",
        # Process Conditions
        "pressure", "temperature", "flow", "level", "reaction", "runaway", "composition", "contamination", "blockage", "plug", "choke", "freeze", "vibration", "noise", "surge", "hammer", "cavitation", "trip", "shutdown", "startup", "upset"
    ],

    # --- 4. HUMAN & ORGANIZATION POOL ---
    "Human_Org": [
        "operator", "personnel", "staff", "technician", "engineer", "contractor", "driver", "human", "error", "mistake", "slip", "lapse", "violation",
        "procedure", "instruction", "manual", "permit", "ptw", "isolation", "loto", "tagout", "handover", "communication", "shift",
        "maintenance", "inspection", "testing", "calibration", "repair", "modification", "design", "construction", "commissioning", "training", "competence", "supervision", "management", "culture"
    ],

    # --- 5. LOCATION / UNIT POOL ---
    "Location_Unit": [
        "plant", "unit", "area", "section", "train", "system", "building", "room", "control room", "laboratory", "warehouse", "storage", "jetty", "berth", "road", "site", "facility"
    ],

    # --- 6. ENVIRONMENT POOL ---
    "Environment": [
        "weather", "storm", "rain", "flood", "wind", "lightning", "earthquake", "tsunami", "external", "third party", "vehicle", "ship", "truck"
    ]
}

def clean_path(p):
    return str(p).replace("/", " > ").replace(" >> ", " > ").strip()

def classify_node(path):
    p = clean_path(path)
    parts = [x.strip() for x in p.split(">")]
    
    # We classify based on the LEAF (the specific item)
    leaf = parts[-1]
    leaf_lower = leaf.lower()
    
    # Check against our Master Map
    for pool, keywords in MASTER_MAP.items():
        for kw in keywords:
            # Word boundary check (matches "oil" but not "boil")
            if re.search(r'\b' + re.escape(kw) + r'\b', leaf_lower):
                
                # Check if the path ALREADY starts with this Pool
                # e.g. "Equipment > Pumps" -> Don't change it to "Equipment > Equipment > Pumps"
                if parts[0] == pool:
                    return p
                
                # If the path is just the leaf (e.g. "Pump"), put it in the Pool
                if len(parts) == 1:
                    return f"{pool} > {leaf}"
                
                # If the path has structure (e.g. "Pumps > Centrifugal"), put the Pool at the top
                # Result: "Equipment > Pumps > Centrifugal"
                
                # Heuristic: If the current root is NOT one of our pools, replace it or prepend?
                # Let's PREPEND the pool to group everything together.
                return f"{pool} > {p}"

    # If no match, it remains an "Individual" (Unclassified relative to pools)
    return p # Or return f"Individuals > {p}" if you want to group them too

def run():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"Error: {e}")
        return

    path_col = next((c for c in df.columns if "path" in c.lower()), "Taxonomy_Path")
    print(f"Processing column: {path_col}")
    
    # 1. Apply Classification
    original_paths = df[path_col].tolist()
    df[path_col] = df[path_col].apply(classify_node)
    new_paths = df[path_col].tolist()
    
    # 2. Identify "Individuals" (Nodes that didn't get moved into a Pool)
    # We check if the new path starts with one of our Pool names
    pool_names = list(MASTER_MAP.keys())
    
    individuals = []
    for p in new_paths:
        root = p.split(">")[0].strip()
        if root not in pool_names:
            individuals.append(p)
            
    # 3. Report Results
    unique_inds = sorted(list(set(individuals)))
    print("-" * 30)
    print(f"CLASSIFICATION COMPLETE")
    print(f"Total Rows: {len(df)}")
    print(f"Classified into Pools: {len(df) - len(individuals)}")
    print(f"Remaining Individuals: {len(individuals)}")
    print("-" * 30)
    print(f"TOP 20 UNCLASSIFIED INDIVIDUALS (Examples):")
    for i in unique_inds[:20]:
        print(f" - {i}")
    print("-" * 30)
    
    # 4. Save
    df.to_excel(OUTPUT_FILE, sheet_name=SHEET_NAME, index=False)
    print(f"Saved grouped taxonomy to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run()