const API_BASE = 'http://localhost:5000';

// --- Setup Wizard Logic ---
let lotCount = 0;

function addLotRow() {
    const container = document.getElementById('lots-config-container');
    const rowId = lotCount++;
    
    const html = `
        <div class="bg-slate-900/50 p-6 rounded-xl border border-slate-700 relative fade-in" id="row-${rowId}">
            <div class="absolute top-4 right-4">
                <button type="button" onclick="removeLotRow(${rowId})" class="text-slate-500 hover:text-red-400 transition-colors" ${rowId === 0 ? 'disabled style="opacity:0.3"' : ''}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
                </button>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="col-span-2">
                    <label class="block text-sm font-medium text-slate-400 mb-1">Lot Name</label>
                    <input type="text" name="lot_name_${rowId}" value="Parking Lot ${rowId + 1}" class="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-emerald-500 focus:outline-none">
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-slate-400 mb-1">Mask Image</label>
                    <input type="file" name="mask_file_${rowId}" accept="image/*" required class="w-full bg-slate-800 border border-slate-600 rounded-lg text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-emerald-600 file:text-white hover:file:bg-emerald-500">
                </div>

                <div>
                     <label class="block text-sm font-medium text-slate-400 mb-1">Video Source</label>
                     <select onchange="toggleVideoInput(${rowId}, this.value)" name="video_type_${rowId}" class="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-white mb-2">
                        <option value="file">Upload Video File</option>
                        <option value="url">RTSP Stream / URL / Local Path</option>
                     </select>
                     
                     <input type="file" id="vid-file-${rowId}" name="video_file_${rowId}" accept="video/*" class="w-full bg-slate-800 border border-slate-600 rounded-lg text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500">
                     
                     <input type="text" id="vid-url-${rowId}" name="video_url_${rowId}" placeholder="rtsp://... or 0 for Webcam" class="hidden w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
            </div>
        </div>
    `;
    container.insertAdjacentHTML('beforeend', html);
}

function removeLotRow(id) {
    document.getElementById(`row-${id}`).remove();
}

function toggleVideoInput(id, type) {
    const fileInput = document.getElementById(`vid-file-${id}`);
    const urlInput = document.getElementById(`vid-url-${id}`);
    if (type === 'file') {
        fileInput.classList.remove('hidden');
        urlInput.classList.add('hidden');
    } else {
        fileInput.classList.add('hidden');
        urlInput.classList.remove('hidden');
    }
}

// --- System Initialization ---
async function checkStatus() {
    try {
        const res = await fetch(`${API_BASE}/system_status`);
        const data = await res.json();
        if (data.configured && data.running) {
            showDashboard();
        } else {
            document.getElementById('setup-wizard').classList.remove('hidden');
            if (document.getElementById('lots-config-container').children.length === 0) {
                addLotRow();
            }
        }
    } catch (e) {
        console.error("Connection error", e);
    }
}

document.getElementById('config-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('start-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = `<div class="spinner mr-2"></div> Starting...`;
    btn.disabled = true;

    const formData = new FormData(e.target);
    
    try {
        const res = await fetch(`${API_BASE}/configure`, { method: 'POST', body: formData });
        const result = await res.json();
        
        if (result.success) {
            document.getElementById('setup-wizard').classList.add('hidden');
            showDashboard();
        } else {
            alert('Error: ' + result.message);
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    } catch (err) {
        alert('Connection Failed: ' + err.message);
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});

function showDashboard() {
    document.getElementById('dashboard').classList.remove('hidden');
    document.getElementById('video-feed').src = `${API_BASE}/video_feed`;
    
    // Reduced polling rate slightly to ease load, but updates will be smooth
    setInterval(updateData, 2000);
    setInterval(updateHistory, 2500);
    updateData();
    updateHistory();
}

// --- Dashboard Smart Updating (Fixed Glitching) ---
async function updateData() {
    try {
        const res = await fetch(`${API_BASE}/data`);
        const lots = await res.json();
        const container = document.getElementById('stats-container');

        if (container.children.length === 1 && container.children[0].innerText.includes("Waiting")) {
            container.innerHTML = "";
        }

        lots.forEach(lot => {
            let card = document.getElementById(`lot-card-${lot.lot_id}`);
            const percentFull = lot.total > 0 ? Math.round((lot.occupied / lot.total) * 100) : 0;
            const available = Math.max(0, lot.total - lot.occupied);
            
            if (!card) {
                const html = `
                    <div id="lot-card-${lot.lot_id}" class="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-lg fade-in">
                        <div class="flex justify-between items-start mb-4">
                            <div>
                                <h3 class="text-lg font-bold text-white">${lot.lot_name}</h3>
                                <p class="text-xs text-slate-400">ID: ${lot.lot_id}</p>
                            </div>
                            <span id="badge-${lot.lot_id}" class="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400">
                                ${percentFull}% Full
                            </span>
                        </div>
                        <div class="flex items-end justify-between">
                            <div>
                                <span id="avail-${lot.lot_id}" class="text-4xl font-bold text-emerald-400">${available}</span>
                                <span class="text-sm text-slate-400 ml-1">spots free</span>
                            </div>
                            <div class="text-right">
                                <div class="text-2xl font-bold text-slate-200" id="occ-${lot.lot_id}">${lot.occupied}</div>
                                <div class="text-xs text-slate-500">Occupied</div>
                            </div>
                        </div>
                        <div class="mt-4 h-2 bg-slate-700 rounded-full overflow-hidden">
                            <div id="bar-${lot.lot_id}" class="h-full bg-emerald-500 transition-all duration-500" style="width: ${percentFull}%"></div>
                        </div>
                    </div>
                `;
                container.insertAdjacentHTML('beforeend', html);
            } else {
                document.getElementById(`avail-${lot.lot_id}`).innerText = available;
                document.getElementById(`occ-${lot.lot_id}`).innerText = lot.occupied;
                document.getElementById(`badge-${lot.lot_id}`).innerText = `${percentFull}% Full`;
                document.getElementById(`bar-${lot.lot_id}`).style.width = `${percentFull}%`;
                
                const bar = document.getElementById(`bar-${lot.lot_id}`);
                if (percentFull > 90) bar.className = "h-full bg-red-500 transition-all duration-500";
                else if (percentFull > 70) bar.className = "h-full bg-yellow-500 transition-all duration-500";
                else bar.className = "h-full bg-emerald-500 transition-all duration-500";
            }
        });
    } catch (e) { console.error(e); }
}

// --- Smart History Update (DOM Diffing to Fix Flashing) ---
async function updateHistory() {
    try {
        const res = await fetch(`${API_BASE}/parking_history`);
        const history = await res.json();
        const list = document.getElementById('history-list');

        if (history.length === 0 && list.children.length === 0) {
            list.innerHTML = `<div class="text-center text-slate-500 text-sm py-4">No activity recorded yet</div>`;
            return;
        }

        // 1. Identify which IDs are currently in the new data
        const newIds = new Set(history.map(item => item.unique_id || `temp-${item.timestamp_in}-${item.spot_id}`));

        // 2. Remove items from DOM that are no longer in the data (or beyond limit)
        Array.from(list.children).forEach(child => {
            // Ignore the "No activity" message div
            if (child.id && !newIds.has(child.id)) {
                child.remove();
            }
        });

        // 3. Add or Update items
        history.forEach((item, index) => {
            const uniqueId = item.unique_id || `temp-${item.timestamp_in}-${item.spot_id}`;
            let existingEl = document.getElementById(uniqueId);

            const isParked = item.is_active;
            const iconColor = isParked ? 'text-emerald-400' : 'text-slate-400';
            const borderColor = isParked ? 'border-emerald-500/20' : 'border-slate-700';
            const bg = isParked ? 'bg-emerald-500/5' : 'bg-slate-800';
            
            let plateDisplay = item.plate_number;
            if(plateDisplay === "Waiting...") plateDisplay = `<span class="animate-pulse text-yellow-400">Scanning...</span>`;
            if(item.processing_status === 'queued') plateDisplay = `<span class="animate-pulse text-indigo-400">Processing...</span>`;

            // Screenshot Button Logic
            let viewBtn = '';
            if (item.plate_image && item.plate_image !== 'None' && item.plate_image !== 'N/A') {
                viewBtn = `
                    <button onclick="openImageModal('${item.plate_image}', '${item.plate_number}')" 
                            class="ml-2 px-2 py-1 text-xs font-semibold bg-indigo-500/20 text-indigo-300 rounded hover:bg-indigo-500/30 transition-colors flex items-center gap-1 border border-indigo-500/30">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
                        View
                    </button>
                `;
            }

            const innerHTMLContent = `
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-3">
                        <div class="p-2 rounded-lg bg-slate-900 ${iconColor}">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><path d="M9 17h6"/><circle cx="17" cy="17" r="2"/></svg>
                        </div>
                        <div>
                            <div class="text-sm font-bold text-white flex items-center">
                                ${plateDisplay}
                                ${viewBtn}
                            </div>
                            <div class="text-xs text-slate-400">${item.vehicle_type} • ${item.color}</div>
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="text-xs font-bold text-slate-300">Spot ${item.spot_id}</div>
                        <div class="text-xs text-slate-500">${item.timestamp_in.split(' ')[0]}</div>
                    </div>
                </div>
            `;

            if (!existingEl) {
                // Create New Element
                const newEl = document.createElement('div');
                newEl.id = uniqueId;
                newEl.className = `p-3 rounded-xl border ${borderColor} ${bg} list-item-enter`;
                newEl.innerHTML = innerHTMLContent;
                
                // Insert at correct position (usually top)
                if (index === 0) {
                    list.prepend(newEl);
                } else {
                    // Find the element before this one in the new array
                    const prevItem = history[index-1];
                    const prevEl = document.getElementById(prevItem.unique_id);
                    if (prevEl) {
                        prevEl.after(newEl);
                    } else {
                        list.appendChild(newEl);
                    }
                }
            } else {
                // Update existing element
                if (!existingEl.innerHTML.includes(plateDisplay)) {
                     existingEl.innerHTML = innerHTMLContent;
                     existingEl.className = `p-3 rounded-xl border ${borderColor} ${bg}`; // Remove animation class on update
                }
            }
        });

        // 4. Client-side limit enforcement
        while (list.children.length > 100) {
            list.lastElementChild.remove();
        }

    } catch (e) { console.error(e); }
}

// --- Image Modal Logic ---
function openImageModal(filename, plate) {
    const modal = document.getElementById('image-modal');
    const img = document.getElementById('modal-img-content');
    const cap = document.getElementById('modal-caption');
    
    img.src = `${API_BASE}/plate_screenshots/${filename}`;
    cap.innerText = `Plate: ${plate}`;
    
    modal.classList.remove('hidden');
    // Force reflow
    void modal.offsetWidth;
    modal.classList.remove('opacity-0');
}

function closeImageModal() {
    const modal = document.getElementById('image-modal');
    modal.classList.add('opacity-0');
    setTimeout(() => {
        modal.classList.add('hidden');
        document.getElementById('modal-img-content').src = ''; // clear memory
    }, 300);
}

// Init
checkStatus();
