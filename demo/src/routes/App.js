import '../App.css';
import {results} from "../data-loader";
import {ENGLISH_EXERCISE_NAMES_MAP} from "../constants";

function App() {
    return (
        <div className="App">
            {Object.entries(results).map(([exercise, exercise_results]) =>
                <div key={exercise}>
                    <h3>{ENGLISH_EXERCISE_NAMES_MAP[exercise]}</h3>
                    <ol className="d-flex flex-column align-items-center">
                        {Object.entries(exercise_results["file_order"]).map(([file_index, file]) =>
                            <li key={`${exercise}-${file_index}`}>
                                <a href={`/review?exercise=${exercise}&id=${file_index}`}>{file}</a>
                            </li>
                        )}
                    </ol>
                </div>
            )}
        </div>
    );
}

export default App;
