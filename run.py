import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    # Ensure upload directories exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('plate_screenshots', exist_ok=True)
    
    print("\n" + "=" * 50)
    print("ParkAlisto - Multi-Lot Parking Analytics System")
    print("=" * 50)
    print("✓ System Initialized")
    print("✓ Access Dashboard: http://localhost:5000")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
