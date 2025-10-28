export const ctx = document.getElementById('metricsChart').getContext('2d');
export const metricsChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        datasets: []
    },
    options: {
        responsive: true,
        plugins: {
            legend: { position: 'top' },
            tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.raw.toFixed(2)}%` } }
        },
        scales: { y: { beginAtZero: true, ticks: { callback: (v) => `${v}%` } } }
    }
});

export function updateChart(metricsData) {
    if (metricsData) {
        const datasets = [];
        const colors = {
            baseline: 'rgba(54, 162, 235, 0.7)',
            random_forest: 'rgba(255, 159, 64, 0.7)'
        };
        for (const modelKey in metricsData) {
            const modelMetrics = metricsData[modelKey];
            datasets.push({
                label: modelKey,
                data: [
                    modelMetrics.accuracy * 100,
                    modelMetrics.precision * 100,
                    modelMetrics.recall * 100,
                    modelMetrics.f1 * 100
                ],
                backgroundColor: colors[modelKey] || 'rgba(255, 99, 132, 0.7)',
            });
        }
        metricsChart.data.datasets = datasets;
        metricsChart.update();
    }
}