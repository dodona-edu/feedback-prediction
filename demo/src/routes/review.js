import React from "react";
import CodeView from "../components/code-view";
import {useQuery} from "../util"
import {results} from "../data-loader";

function Review() {
    const params = useQuery();
    const exercise = params.get("exercise");
    const file_index = Number(params.get("id"));

    const fileOrder = results[exercise]["file_order"];

    const file = fileOrder[file_index];
    const filePath = "/exercises/" + exercise + "/" + file;

    const fileResults = results[exercise]["results"][file];

    return (
        <div>
            <h3 className={"text-center"}>Submission file {file_index + 1}: {file}</h3>
            <CodeView filePath={filePath} fileResults={fileResults}></CodeView>
            <div className={"text-center mb-3"}>
                {file_index > 0 ?
                    <a href={`/review?exercise=${exercise}&id=${file_index - 1}`} className={"btn btn-primary"}>{"<"} Previous submission</a>
                    : ""
                }
                <a href={"/"} className={"btn btn-secondary ms-2 me-2"}>Home</a>
                {file_index < fileOrder.length - 1 ?
                    <a href={`/review?exercise=${exercise}&id=${file_index + 1}`} className={"btn btn-primary"}>Next submission {">"}</a>
                    : ""
                }
            </div>
        </div>
    )
}

export default Review;