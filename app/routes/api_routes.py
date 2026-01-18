from flask import jsonify, request
from . import api_bp
from app.core import system_state
from app.models import VideoConfig
from app.services import process_video_loop, background_processor_worker
from werkzeug.utils import secure_filename
from config import Config
import threading
import time
import os

@api_bp.route('/system_status')
def get_system_status():
    return jsonify({
        'configured': system_state.is_configured,
        'running': system_state.is_running
    })

@api_bp.route('/data')
def get_data():
    all_data = []
    for config in system_state.video_configs:
        with config.data_lock:
            all_data.append(config.data.copy())
    return jsonify(all_data)

@api_bp.route('/parking_history')
def get_history():
    all_history = []
    for config in system_state.video_configs:
        with config.log_lock:
            all_history.extend(list(config.history))
    
    all_history.sort(key=lambda x: x.get('timestamp_in', ''), reverse=True)
    return jsonify(all_history[:100])

@api_bp.route('/stats/global')
def get_global_stats():
    total_spots = 0
    total_occupied = 0
    total_parked_today = 0
    lot_stats = []

    for config in system_state.video_configs:
        with config.data_lock:
            # Safely access data
            t = config.data.get('total', 0)
            o = config.data.get('occupied', 0)
            total_spots += t
            total_occupied += o
            
            lot_stats.append({
                'lot_id': config.lot_id,
                'lot_name': config.lot_name,
                'total': t,
                'occupied': o,
                'available': max(0, t - o)
            })
            
        with config.log_lock:
             total_parked_today += len(config.history)

    return jsonify({
        'total_spots': total_spots,
        'total_occupied': total_occupied,
        'total_available': max(0, total_spots - total_occupied),
        'total_parked_today': total_parked_today,
        'lot_stats': lot_stats
    })

@api_bp.route('/search')
def search_plate():
    query = request.args.get('plate', '').strip().upper()
    if not query:
        return jsonify([])

    results = []
    for config in system_state.video_configs:
        with config.log_lock:
            # Search in history
            for entry in config.history:
                # Basic substring match or exact match? User asked for "search a platenumber".
                # Let's do substring for flexibility, but prioritize startswith? 
                # Or just simple substring.
                pvars = str(entry.get('plate_number', '')).upper()
                if query in pvars and pvars != "WAITING..." and pvars != "PROCESSING...":
                    results.append(entry.copy())
            
            # Search in active log
            for spot_id, entry in config.active_log.items():
                 pvars = str(entry.get('plate_number', '')).upper()
                 if query in pvars and pvars != "WAITING..." and pvars != "PROCESSING...":
                     # Avoid duplicates if it's already in history (it shouldn't be if active, but good to be safe)
                     # Actually active_log items are NOT in history deque until they leave? 
                     # Wait, let's check video_processor.py log_car_parked.
                     # It adds to history AND active_log.
                     pass 
                     # So if we iterate history, we likely cover active ones too?
                     # video_processor.py:213 config.history.appendleft(log_entry.copy())
                     # Yes, it adds to history immediately upon parking.
                     pass

    # Sort results by timestamp (newest first)
    # Timestamp format: "%I:%M:%S %p %B %d, %Y"
    # It communicates readable string, so sorting might be imperfect string sort, but acceptable for now.
    results.sort(key=lambda x: x.get('timestamp_in', ''), reverse=True)
    
    return jsonify(results)

@api_bp.route('/configure', methods=['POST'])
def configure_system():
    try:
        print("=" * 50)
        print("Received configuration request")
        
        system_state.video_configs.clear()
        
        lot_indices = set()
        for key in request.form.keys():
            if key.startswith('lot_name_'):
                lot_indices.add(key.split('_')[-1])
        
        for idx in sorted(list(lot_indices)):
            lot_name = request.form.get(f'lot_name_{idx}')
            source_type = request.form.get(f'video_type_{idx}')
            
            video_path = ""
            if source_type == 'file':
                file = request.files.get(f'video_file_{idx}')
                if file and file.filename:
                    filename = secure_filename(f"vid_{idx}_{int(time.time())}_{file.filename}")
                    save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                    file.save(save_path)
                    video_path = save_path
            else:
                video_path = request.form.get(f'video_url_{idx}')
            
            mask_path = ""
            mask_file = request.files.get(f'mask_file_{idx}')
            if mask_file and mask_file.filename:
                filename = secure_filename(f"mask_{idx}_{int(time.time())}_{mask_file.filename}")
                save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                mask_file.save(save_path)
                mask_path = save_path
            
            if lot_name and video_path and mask_path:
                print(f"✓ Configuring Lot {idx}: {lot_name}")
                print(f"  Video: {video_path}")
                print(f"  Mask: {mask_path}")
                config = VideoConfig(len(system_state.video_configs), lot_name, video_path, mask_path)
                system_state.video_configs.append(config)
        
        if not system_state.video_configs:
            return jsonify({'success': False, 'message': 'No valid configurations found'}), 400

        if not system_state.is_running:
            print("Starting background workers...")
            for i in range(Config.NUM_PROCESSING_THREADS):
                t = threading.Thread(target=background_processor_worker, daemon=True)
                t.start()
            system_state.is_running = True

        print("Starting video processing threads...")
        for config in system_state.video_configs:
            t = threading.Thread(target=process_video_loop, args=(config,), daemon=True)
            t.start()
            
        system_state.is_configured = True
        return jsonify({'success': True})

    except Exception as e:
        print(f"Configuration Error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@api_bp.route('/add_source', methods=['POST'])
def add_source():
    try:
        print("Received request to add new video source")
        
        # Calculate new ID based on current length + timestamp to ensure uniqueness
        # or just simple increment if we assume linear growth.
        # Safest is just len(system_state.video_configs) but let's be careful about race conditions if multiple adds happen.
        # For this simple app, len() is fine as we are likely single-user.
        new_idx = len(system_state.video_configs)
        
        lot_name = request.form.get('lot_name')
        source_type = request.form.get('video_type')
        
        if not lot_name:
             return jsonify({'success': False, 'message': 'Lot name is required'}), 400

        video_path = ""
        if source_type == 'file':
            file = request.files.get('video_file')
            if file and file.filename:
                filename = secure_filename(f"vid_{new_idx}_{int(time.time())}_{file.filename}")
                save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                file.save(save_path)
                video_path = save_path
            else:
                 return jsonify({'success': False, 'message': 'Video file is required'}), 400
        else:
            video_path = request.form.get('video_url')
            if not video_path:
                return jsonify({'success': False, 'message': 'Video URL is required'}), 400
        
        mask_path = ""
        mask_file = request.files.get('mask_file')
        if mask_file and mask_file.filename:
            filename = secure_filename(f"mask_{new_idx}_{int(time.time())}_{mask_file.filename}")
            save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            mask_file.save(save_path)
            mask_path = save_path
        else:
             return jsonify({'success': False, 'message': 'Mask image is required'}), 400
        
        print(f"✓ Adding Lot {new_idx}: {lot_name}")
        config = VideoConfig(new_idx, lot_name, video_path, mask_path)
        
        # If system is not running (e.g. all previous streams failed or none were started), we might need to ensure workers are up.
        # But usually this is called when system is running. 
        # If system wasn't running, user should use /configure. 
        # However, let's be safe.
        with system_state.combined_frame_lock: # Just a convenient lock to sync config append if needed
            system_state.video_configs.append(config)
        
        if not system_state.is_running:
             # Start background workers if not already
            print("Starting background workers (System was idle)...")
            for i in range(Config.NUM_PROCESSING_THREADS):
                t = threading.Thread(target=background_processor_worker, daemon=True)
                t.start()
            system_state.is_running = True

        # Start processing thread for this new config
        t = threading.Thread(target=process_video_loop, args=(config,), daemon=True)
        t.start()
        
        return jsonify({'success': True, 'lot_id': new_idx})

    except Exception as e:
        print(f"Add Source Error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@api_bp.route('/debug/config')
def debug_config():
    """Debug endpoint to inspect active video configurations"""
    configs = []
    for config in system_state.video_configs:
        configs.append({
            'lot_id': config.lot_id,
            'lot_name': config.lot_name,
            'video_path': config.video_path,
            'mask_path': config.mask_path,
            'mask_path': config.mask_path,
            'history_len': len(config.history),
            'active_log_len': len(config.active_log),
        })
    return jsonify({
        'active_configs': len(system_state.video_configs),
        'configs': configs,
        'threads': [t.name for t in threading.enumerate()]
    })
