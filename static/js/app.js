import { fetchData } from './data.js';
import { map, trueFiresLayer, lgbmLayer, rfLayer, addMarkersToLayer } from './map.js';
import { metricsChart, updateChart } from './chart.js';

async function loadAllData() {
    // Load metrics
    const metricsData = await fetchData('static/metrics.json');
    updateChart(metricsData);

    // Load map data
    const trueFiresData = await fetchData('data/true_fires.geojson');
    if (trueFiresData.features) {
         addMarkersToLayer(trueFiresLayer, trueFiresData.features, 'img/icon_red.png', 
            item => `<b>실제 산불</b><br>주소: ${item.properties.full_address || 'N/A'}`
        );
    }

    const lgbmData = await fetchData('data/refined_predicted_fire_markers.json');
    const damageData = await fetchData('data/damage_predictions.json');
    const damageMap = {};
    if (damageData) {
        damageData.forEach(item => {
            const key = `${item.lat.toFixed(6)},${item.lon.toFixed(6)}`;
            damageMap[key] = item.predicted_damage;
        });
    }

    addMarkersToLayer(lgbmLayer, lgbmData, 'img/icon_blue.png', 
        item => {
            const key = `${parseFloat(item.lat).toFixed(6)},${parseFloat(item.lon).toFixed(6)}`;
            const damage = damageMap[key];
            let popupContent = `<b>LGBM 예측</b><br>확률: ${item.probability ? item.probability.toFixed(2) : 'N/A'}`;
            if (damage !== undefined) {
                popupContent += `<br>예상 피해 면적: ${damage.toFixed(4)} ha`;
            }
            return popupContent;
        },
        true,
        damageMap
    );

    const rfData = await fetchData('data/rf_predict.csv');
    addMarkersToLayer(rfLayer, rfData, 'img/icon_orange.png', 
        item => {
            const key = `${parseFloat(item.LAT).toFixed(6)},${parseFloat(item.LON).toFixed(6)}`;
            const damage = damageMap[key];
            let popupContent = `<b>RF 예측</b><br>확률: ${item.FIRE_PROBABILITY ? parseFloat(item.FIRE_PROBABILITY).toFixed(2) : 'N/A'}`;
            if (damage !== undefined) {
                popupContent += `<br>예상 피해 면적: ${damage.toFixed(4)} ha`;
            }
            return popupContent;
        },
        true,
        damageMap
    );
}

// --- Filtering Logic ---
document.getElementById('filter-button').addEventListener('click', () => {
    const startDate = document.getElementById('start-date').valueAsDate;
    const endDate = document.getElementById('end-date').valueAsDate;

    if (!startDate || !endDate) {
        alert('시작 날짜와 종료 날짜를 모두 선택해주세요.');
        return;
    }
    endDate.setHours(23, 59, 59, 999);

    [trueFiresLayer, lgbmLayer, rfLayer].forEach(layer => {
        let visibleMarkers = [];
        layer.eachLayer(marker => {
            if (marker.fireDate) {
                 if (marker.fireDate >= startDate && marker.fireDate <= endDate) {
                    visibleMarkers.push(marker);
                }
            }
        });
        layer.clearLayers();
        layer.addLayers(visibleMarkers);
    });
});

// Handle URL parameters
window.addEventListener('load', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const lat = urlParams.get('lat');
    const lon = urlParams.get('lon');

    if (lat && lon) {
        map.setView([lat, lon], 13);
        L.circle([lat, lon], {
            color: 'red',
            fillColor: '#f03',
            fillOpacity: 0.5,
            radius: 1000 // 1km radius
        }).addTo(map);
    }
});

loadAllData();