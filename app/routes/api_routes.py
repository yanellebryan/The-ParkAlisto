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
