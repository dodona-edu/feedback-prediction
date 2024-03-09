import SyntaxHighlighter, { createElement} from 'react-syntax-highlighter';
import {idea} from "react-syntax-highlighter/dist/cjs/styles/hljs";
import { useEffect, useState} from "react";
import {load_file} from "../data-loader";
import FeedbackBox from "./feedback-box"

function virtualizedRenderer(fileResults) {
    return ({rows, stylesheet, useInlineStyles}) => {
            let count = 0;
            return (
                <div>
                    {rows.map((row, index) => {
                        const key = `code-row-${index}`;
                        const code = createElement({
                            node: row,
                            stylesheet,
                            useInlineStyles,
                            key,
                        });
                        if (index in fileResults) {
                            const lineResults = fileResults[index];
                            const currentCount = count;
                            count += 1;
                            return (
                                <div key={key} id={"feedback-" + currentCount}>
                                    {code}
                                    <FeedbackBox actualMessages={lineResults['actual']} predictedMessages={lineResults['predicted']}></FeedbackBox>
                                </div>
                            )
                        } else {
                            return (
                                <div key={key}>
                                    {code}
                                </div>
                            )
                        }
                    })}
                </div>
        );
    }
}

function CodeView({filePath, fileResults}) {
    const [codeString, setCodeString] = useState("");

    useEffect(() => {
        load_file(filePath).then(code => setCodeString(code)).catch(er => console.log(er));
    }, [filePath]);

    return (
        <SyntaxHighlighter language="python" style={idea} showLineNumbers={true} wrapLines={true} wrapLongLines={true} lineProps={(nr) => {
            return {id: nr}
        }} renderer={virtualizedRenderer(fileResults)}>
            {codeString}
        </SyntaxHighlighter>
    );
}

export default CodeView
