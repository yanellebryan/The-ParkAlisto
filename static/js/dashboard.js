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
                    <i data-lucide="trash-2" class="w-5 h-5"></i>
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
    lucide.createIcons();
}

function removeLotRow(id) {
    document.getElementById(`row-${id}`).remove();
}

// Explicitly attach to window to ensure access from inline HTML handler
window.toggleVideoInput = function (id, type) {
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
    // document.getElementById('video-feed').src = `${API_BASE}/video_feed`; // REMOVED legacy single feed

    // Reduced polling rate slightly to ease load, but updates will be smooth
    setInterval(updateData, 2000);
    setInterval(updateHistory, 2500);
    updateData();
    updateHistory();
    lucide.createIcons();
}

// --- Dashboard Smart Updating (Fixed Glitching) ---
async function updateData() {
    try {
        const res = await fetch(`${API_BASE}/data`);
        const lots = await res.json();
        const container = document.getElementById('stats-container');

        // Render/Update Video Grid
        renderVideoGrid(lots);

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

        const newIds = new Set(history.map(item => item.unique_id || `temp-${item.timestamp_in}-${item.spot_id}`));

        Array.from(list.children).forEach(child => {
            // Remove if not in new data OR if it's the "No activity" placeholder (which has no ID)
            if ((child.id && !newIds.has(child.id)) || !child.id) {
                child.remove();
            }
        });

        history.forEach((item, index) => {
            const uniqueId = item.unique_id || `temp-${item.timestamp_in}-${item.spot_id}`;
            let existingEl = document.getElementById(uniqueId);

            // Parse Time
            let timeDisplay = item.timestamp_in;
            try {
                // Backend Format: "%I:%M:%S %p %B %d, %Y" -> "04:45:00 PM January 18, 2026"
                let parts = timeDisplay.split(' ');
                if (parts.length >= 2) {
                    let timePart = parts[0]; // "04:45:00"
                    let ampmPart = parts[1]; // "PM"

                    // Extract HH:MM
                    let hhmm = timePart.substring(0, 5);
                    timeDisplay = `${hhmm} ${ampmPart}`;
                }
            } catch (e) {
                console.error("Time Parse Error", e);
                // Fallback: just show first part
                timeDisplay = item.timestamp_in.split(' ')[0];
            }

            const isParked = item.is_active;
            const statusColor = isParked ? 'text-emerald-400' : 'text-slate-500';
            const statusBg = isParked ? 'bg-emerald-500/10' : 'bg-slate-700/30';
            const statusBorder = isParked ? 'border-emerald-500/20' : 'border-dashed border-slate-700';
            const iconName = isParked ? 'arrow-right-circle' : 'check-circle-2';

            let plateDisplay = item.plate_number;
            if (plateDisplay === "Waiting...") plateDisplay = `<span class="animate-pulse text-yellow-400 text-xs">SCANNING...</span>`;
            if (item.processing_status === 'queued') plateDisplay = `<span class="animate-pulse text-indigo-400 text-xs">PROCESSING...</span>`;

            let viewBtn = '';
            if (item.plate_image && item.plate_image !== 'None' && item.plate_image !== 'N/A') {
                viewBtn = `
                    <button onclick="openImageModal('${item.plate_image}', '${item.plate_number}')" 
                            class="p-1.5 hover:bg-slate-700 rounded-md text-slate-400 hover:text-indigo-400 transition-colors" title="View Snapshot">
                        <i data-lucide="image" class="w-4 h-4"></i>
                    </button>
                `;
            }

            const innerHTMLContent = `
                <div class="flex items-center gap-3">
                    <div class="flex flex-col items-center min-w-[50px]">
                        <span class="text-sm font-bold text-slate-300 font-mono">${timeDisplay}</span>
                        <div class="h-full w-px bg-slate-800 my-1 group-last:hidden"></div>
                    </div>
                    <div class="flex-1 bg-slate-900/50 border ${statusBorder} rounded-xl p-3 flex items-center justify-between hover:bg-slate-800 transition-all group-hover:shadow-md">
                        <div class="flex items-center gap-3">
                            <div class="p-2 rounded-full ${statusBg} ${statusColor}">
                                <i data-lucide="${iconName}" class="w-4 h-4"></i>
                            </div>
                            <div>
                                <div class="text-sm font-bold text-white tracking-wide flex items-center gap-2">
                                    ${plateDisplay}
                                    ${isParked ? '<span class="text-[10px] bg-emerald-500/20 text-emerald-400 px-1.5 rounded uppercase font-bold">In</span>' : ''}
                                </div>
                                <div class="text-xs text-slate-500 flex items-center gap-2">
                                    <span class="uppercase">${item.vehicle_type}</span>
                                    <span>•</span>
                                    <span>${item.color}</span>
                                </div>
                            </div>
                        </div>
                        <div class="text-right flex items-center gap-3">
                             <div class="text-right">
                                <div class="text-xs font-bold text-slate-400 uppercase tracking-wider">Spot ${item.spot_id}</div>
                                <div class="text-[10px] text-slate-600">ID: ${item.lot_id}</div>
                             </div>
                             ${viewBtn}
                        </div>
                    </div>
                </div>
            `;

            if (existingEl) {
                if (existingEl.innerHTML !== innerHTMLContent) {
                    existingEl.innerHTML = innerHTMLContent;
                }
            } else {
                const el = document.createElement('div');
                el.id = uniqueId;
                el.className = "group fade-in";
                el.innerHTML = innerHTMLContent;
                list.appendChild(el);
            }
        });

        lucide.createIcons();
    } catch (e) { console.error(e); }
}

// --- Image Modal Logic ---
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
    setTimeout(() => {
        modal.classList.add('hidden');
        document.getElementById('modal-img-content').src = ''; // clear memory
    }, 300);
}



// --- Add Source Modal Logic ---
function openAddSourceModal() {
    document.getElementById('add-source-modal').classList.remove('hidden');
}

function closeAddSourceModal() {
    document.getElementById('add-source-modal').classList.add('hidden');
    document.getElementById('add-source-form').reset();
    toggleAddSourceVideoInput('file'); // Reset view
}

window.toggleAddSourceVideoInput = function (type) {
    // Backwards compatibility if called directly
    const fileInput = document.getElementById('add-vid-file');
    const urlInput = document.getElementById('add-vid-url');
    if (type === 'file') {
        fileInput.classList.remove('hidden');
        urlInput.classList.add('hidden');
    } else {
        fileInput.classList.add('hidden');
        urlInput.classList.remove('hidden');
    }
}

// Ensure robust event handling
document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('video-type-select'); // Need to ID this element first in HTML if not present, checking...
    // Wait, previous REPLACE failed on HTML too.
    // Let's just rely on the existing window function which IS correct, but maybe cached.
    // The problem might be the HTML select doesn't pass 'this.value' correctly? No that's standard.
    // Re-asserting the window function is fine.
});

document.getElementById('add-source-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('add-source-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = `<div class="spinner mr-2"></div> Adding...`;
    btn.disabled = true;

    const formData = new FormData(e.target);

    try {
        const res = await fetch(`${API_BASE}/add_source`, { method: 'POST', body: formData });
        const result = await res.json();

        if (result.success) {
            closeAddSourceModal();
            // No need to manually refresh, the existing updateData loop will pick up the new lot
            // appearing in the /data endpoint response.
        } else {
            alert('Error: ' + result.message);
        }
    } catch (err) {
        alert('Connection Failed: ' + err.message);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});

// --- Search Logic ---
const searchInput = document.getElementById('search-input');
if (searchInput) {
    searchInput.addEventListener('keypress', async (e) => {
        if (e.key === 'Enter') {
            const query = searchInput.value.trim();
            if (!query) return;

            // Show loading or similar feedback if desired
            searchInput.disabled = true;

            try {
                const res = await fetch(`${API_BASE}/search?plate=${encodeURIComponent(query)}`);
                const results = await res.json();

                openSearchModal(results, query);
            } catch (err) {
                console.error("Search failed", err);
                alert("Search failed. Check console.");
            } finally {
                searchInput.disabled = false;
                searchInput.focus();
            }
        }
    });
}

function openSearchModal(results, query) {
    const modal = document.getElementById('search-modal');
    const list = document.getElementById('search-results-list');
    list.innerHTML = ''; // Clear previous

    if (results.length === 0) {
        list.innerHTML = `
        <div class="text-center py-8 text-slate-400">
                <i data-lucide="search-x" class="w-12 h-12 mx-auto mb-3 opacity-50"></i>
                <p>No results found for "${query}"</p>
            </div>
            `;
    } else {
        results.forEach(item => {
            const timeIn = item.timestamp_in || 'N/A';
            const timeOut = item.timestamp_out || 'Active';
            // Determine if 'Active' means currently parked -> check is_active or timestamp_out
            // If item has no timestamp_out, it's likely still likely there? 
            // The item comes from history which may include active sessions depending on implementation.

            let statusBadge = '<span class="px-2 py-1 bg-emerald-500/20 text-emerald-400 text-xs rounded-full">Parked</span>';
            if (timeOut !== 'Active' && timeOut !== '') {
                statusBadge = '<span class="px-2 py-1 bg-slate-700 text-slate-400 text-xs rounded-full">Departed</span>';
            }

            const html = `
                <div class="bg-slate-900/50 p-4 rounded-xl border border-slate-700 flex flex-col sm:flex-row gap-4">
                    <div class="flex-shrink-0 w-full sm:w-32 h-24 bg-black rounded-lg overflow-hidden border border-slate-600 relative">
                        ${item.plate_image
                    ? `<img src="${API_BASE}/plate_screenshots/${item.plate_image}" class="w-full h-full object-contain cursor-pointer" onclick="openImageModal('${item.plate_image}', '${item.plate_number}')">`
                    : `<div class="flex items-center justify-center h-full text-slate-500 text-xs text-center p-2">No Image</div>`
                }
                    </div>
                    <div class="flex-grow">
                        <div class="flex justify-between items-start mb-2">
                            <div>
                                <h3 class="font-bold text-white text-lg">${item.plate_number}</h3>
                                <div class="text-xs text-slate-400">Lot: ${item.lot_name || 'Total'} | Spot ${item.spot_id}</div>
                            </div>
                            ${statusBadge}
                        </div>
                        <div class="grid grid-cols-2 gap-2 text-sm text-slate-300">
                             <div><span class="text-slate-500">In:</span> ${timeIn}</div>
                             <div><span class="text-slate-500">Out:</span> ${timeOut}</div>
                             <div><span class="text-slate-500">Vehicle:</span> ${item.vehicle_type || 'Unknown'}</div>
                             <div><span class="text-slate-500">Color:</span> ${item.color || 'Unknown'}</div>
                        </div>
                    </div>
                </div>
            `;
            list.insertAdjacentHTML('beforeend', html);
        });
    }

    modal.classList.remove('hidden');
    // Reflow
    void modal.offsetWidth;
    modal.classList.remove('opacity-0');
    lucide.createIcons();
}

function closeSearchModal() {
    const modal = document.getElementById('search-modal');
    modal.classList.add('opacity-0');
    setTimeout(() => {
        modal.classList.add('hidden');
    }, 300);
}

// Init
checkStatus();

// --- Video Grid Logic ---
function renderVideoGrid(lots) {
    const grid = document.getElementById('video-grid');

    // ADJUST GRID LAYOUT DYNAMICALLY
    // If 1 lot, use 1 col. If more, use 2 cols (md+).
    if (lots.length <= 1) {
        grid.className = "flex-1 overflow-y-auto p-4 custom-scrollbar grid grid-cols-1 gap-4";
    } else {
        grid.className = "flex-1 overflow-y-auto p-4 custom-scrollbar grid grid-cols-1 md:grid-cols-2 gap-4";
    }

    // Clear "No active cameras" placeholder if we have lots
    if (lots.length > 0 && grid.children.length === 1 && grid.children[0].innerText.includes("No active")) {
        grid.innerHTML = "";
    }

    lots.forEach(lot => {
        let cell = document.getElementById(`video-cell-${lot.lot_id}`);

        if (!cell) {
            const html = `
        <div id="video-cell-${lot.lot_id}" class="video-cell group" onclick="toggleFullscreen(${lot.lot_id}, event)">
            <img src="${API_BASE}/video_feed/${lot.lot_id}" alt="${lot.lot_name}" loading="lazy" onerror="this.onerror=null; this.src='${API_BASE}/static/images/logo.png'">

            <div class="lot-info-overlay group-hover:opacity-100 transition-opacity">
                <i data-lucide="video" class="w-3 h-3 text-emerald-400"></i>
                <span class="font-mono text-xs text-white tracking-wide">${lot.lot_name}</span>
            </div>
        </div>
            `;
            grid.insertAdjacentHTML('beforeend', html);
            lucide.createIcons();
        }
    });
}

function toggleFullscreen(lotId, event) {
    // If clicking button inside, let it bubble or handle specific?
    // Actually the whole cell click triggers this.

    const cell = document.getElementById(`video - cell - ${lotId}`);
    if (!cell) return;

    // Toggle active fullscreen class
    if (cell.classList.contains('fullscreen-view')) {
        cell.classList.remove('fullscreen-view');
        // Remove close button
        const closeBtn = cell.querySelector('.fullscreen-close');
        if (closeBtn) closeBtn.remove();
    } else {
        cell.classList.add('fullscreen-view');
        // Add close button if not exists
        if (!cell.querySelector('.fullscreen-close')) {
            const closeBtn = document.createElement('div');
            closeBtn.className = 'fullscreen-close';
            closeBtn.innerHTML = '<i data-lucide="x" class="w-6 h-6"></i>';
            closeBtn.onclick = (e) => {
                e.stopPropagation();
                toggleFullscreen(lotId);
            };
            cell.appendChild(closeBtn);
            lucide.createIcons();
        }
    }
}

function toggleGlobalFullscreen() {
    const container = document.getElementById('video-feed-container');
    if (container.classList.contains('fullscreen-view')) {
        container.classList.remove('fullscreen-view');
        // Restore grid layout
        container.style.padding = "";
        const closeBtn = container.querySelector('.global-close');
        if (closeBtn) closeBtn.remove();
    } else {
        container.classList.add('fullscreen-view');
        // Ensure grid takes full space
        container.style.display = "flex";
        container.style.flexDirection = "column";

        const closeBtn = document.createElement('div');
        closeBtn.className = 'fullscreen-close global-close';
        closeBtn.innerHTML = '<i data-lucide="minimize-2" class="w-6 h-6"></i>';
        closeBtn.style.top = "10px";
        closeBtn.style.right = "10px";
        closeBtn.onclick = (e) => {
            e.stopPropagation();
            toggleGlobalFullscreen();
        };
        container.appendChild(closeBtn);
        lucide.createIcons();
    }
}
