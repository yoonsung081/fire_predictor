import { fetchData } from './data.js'; // Assuming fetchData is needed here, or passed in

export const map = L.map('map').setView([36.5, 127.5], 7);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

export const trueFiresLayer = L.markerClusterGroup().addTo(map);
export const lgbmLayer = L.markerClusterGroup().addTo(map);
export const rfLayer = L.markerClusterGroup().addTo(map);

export const layers = {
    "실제 산불": trueFiresLayer,
    "RandomForest 예측": rfLayer,
    "LightGBM 예측": lgbmLayer
};
L.control.layers(null, layers).addTo(map);

let damageCircle = null;

export function drawDamageCircle(lat, lon, damage) {
    if (damageCircle) {
        map.removeLayer(damageCircle);
    }
    if (damage === undefined) return;

    const radius = Math.sqrt(damage * 10000 / Math.PI); // damage is in ha, convert to m^2 to get radius in meters
    damageCircle = L.circle([lat, lon], {
        radius: radius,
        color: 'red',
        fillColor: '#f03',
        fillOpacity: 0.5
    }).addTo(map);
}

export function addMarkersToLayer(layer, data, iconUrl, popupContentFn, isPrediction = false, damageMap = {}) {
    layer.clearLayers();
    data.forEach(item => {
        const lat = parseFloat(item.lat || item.LAT || (item.properties && item.properties.LAT) || item.REFINE_WGS84_LAT);
        const lon = parseFloat(item.lon || item.LON || (item.properties && item.properties.LON) || item.REFINE_WGS84_LOGT);

        if (!lat || !lon) return;

        const marker = L.marker([lat, lon], {
            icon: L.icon({ iconUrl, iconSize: [25, 25] })
        });
        marker.bindPopup(popupContentFn(item));
        
        if (item.properties && item.properties.date) {
            marker.fireDate = new Date(item.properties.date);
        }
        
        if (isPrediction) {
            const key = `${lat.toFixed(6)},${lon.toFixed(6)}`;
            const damage = damageMap[key];
            marker.on('click', () => {
                drawDamageCircle(lat, lon, damage);
            });
        }

        layer.addLayer(marker);
    });
}