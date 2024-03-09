import Papa from "papaparse"

const results = await load_results();

async function load_file(filePath) {
    const file = await fetch(filePath);
    return await file.text();
}

async function load_results() {
    const text = await load_file("results/fitting_results_505886137.csv")
    const { data } = Papa.parse(text, {
        complete(results, _) {
            return results;
        }
    });

    // remove empty row from data
    data.pop();

    const mapping = {};
    for (const line of data) {
        const exercise = line[0];
        const file = line[1];
        const line_nr = line[2];
        const actual_messages = JSON.parse(line[3]);
        const predicted_messages = JSON.parse(line[4]);

        if (!(exercise in mapping)) {
            mapping[exercise] = {
                "results": {},
                "file_order": [],
            };
        }

        if (!(file in mapping[exercise]["results"])) {
            mapping[exercise]["results"][file] = {};
            mapping[exercise]["file_order"].push(file);
        }

        mapping[exercise]["results"][file][line_nr] = {"actual": actual_messages, "predicted": predicted_messages};
    }

    return mapping;
}

export {load_file, load_results, results}

