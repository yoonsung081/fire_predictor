export async function fetchData(url) {
    const response = await fetch(url);
    if (!response.ok) {
        console.error(`Failed to load ${url}`);
        return [];
    }
    if (url.endsWith('.csv')) {
        const csvText = await response.text();
        const lines = csvText.split('\n').filter(line => line.trim() !== '');
        if (lines.length < 2) return [];
        const headers = lines[0].split(',').map(header => header.trim());
        const data = lines.slice(1).map(line => {
            const values = line.split(',');
            const obj = {};
            headers.forEach((header, index) => {
                obj[header] = values[index] ? values[index].trim() : '';
            });
            return obj;
        });
        return data;
    } else {
        return response.json();
    }
}
