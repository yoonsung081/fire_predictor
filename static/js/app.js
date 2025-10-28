import { fetchData } from './data.js';
import { map, trueFiresLayer, lgbmLayer, rfLayer, addMarkersToLayer } from './map.js';
import { metricsChart, updateChart } from './chart.js';

async function loadAllData() {
    // Load metrics
    const metricsData = await fetchData('static/metrics.json');
    updateChart(metricsData);

    // Load map data
    const trueFiresData = await fetchData('data/true_fires.geojson');
    let minDate = new Date();
    let maxDate = new Date(0); // Epoch

    if (trueFiresData.features) {
         addMarkersToLayer(trueFiresLayer, trueFiresData.features, 'img/icon_red.png', 
            item => {
                const year = item.properties['발생일시_년'];
                const month = item.properties['발생일시_월'];
                const day = item.properties['발생일시_일'];
                const fireDate = new Date(year, month - 1, day); // Month is 0-indexed

                if (fireDate < minDate) minDate = fireDate;
                if (fireDate > maxDate) maxDate = fireDate;

                return `<b>실제 산불</b><br>주소: ${item.properties.full_address || 'N/A'}<br>날짜: ${year}-${month}-${day}`;
            }
        );
    }

    // Set min/max dates for date filters
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');

    if (minDate.getTime() !== new Date().getTime() && maxDate.getTime() !== new Date(0).getTime()) {
        startDateInput.setAttribute('min', minDate.toISOString().split('T')[0]);
        startDateInput.setAttribute('value', minDate.toISOString().split('T')[0]); // Set initial value
        endDateInput.setAttribute('max', maxDate.toISOString().split('T')[0]);
        endDateInput.setAttribute('value', maxDate.toISOString().split('T')[0]); // Set initial value
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
            let popupContent = `<b>LGBM 예측</b>`;
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
            let popupContent = `<b>RF 예측</b>`;
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