import json
import os

PRESETS_FILE = r'C:\Users\user\GeoCalibPack\GeoCalib\presets.json'

def migrate():
    if not os.path.exists(PRESETS_FILE):
        print("presets.json not found")
        return

    with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Detect if it's already 3-level
    # 2-level format: {"ARRI Alexa 35": {"modes": [...]}}
    # 3-level format: {"ARRI": {"Alexa 35": {"modes": [...]}}}
    
    is_3_level = False
    for k, v in data.items():
        if isinstance(v, dict) and "modes" not in v:
            is_3_level = True
            break
            
    if is_3_level:
        print("Already migrated.")
        return

    new_data = {}
    for old_key, v in data.items():
        if "Sony" in old_key:
            b, m = "Sony", old_key.replace("Sony ", "")
        elif "ARRI" in old_key:
            b, m = "ARRI", old_key.replace("ARRI ", "")
        elif "DJI" in old_key:
            b, m = "DJI", old_key.replace("DJI ", "")
        elif "Full Frame" in old_key:
            b, m = "Custom", "Full Frame (35mm)"
        else:
            parts = old_key.split(" ", 1)
            b = parts[0]
            m = parts[1] if len(parts) > 1 else "Unknown Model"
            
        if b not in new_data:
            new_data[b] = {}
        new_data[b][m] = v
        
    # Backup
    import shutil
    shutil.copy2(PRESETS_FILE, PRESETS_FILE + ".v2.bak")
    
    with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
        
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
