function FeedbackBox({actualMessages, predictedMessages}) {
    return (
        <div className={"alert alert-light m-0"}>
            {
                actualMessages.map((m, m_i) => {
                    return <div key={m_i} className={"text-success border-start border-success border-3 py-1 px-3"}>{m}</div>
                })
            }
            {
                predictedMessages.map((m, m_i) => {
                    return <div key={m_i} className={"text-danger border-start border-danger border-3 py-1 px-3 " + (actualMessages.length ? "mt-3" : "")}>{m}</div>
                })
            }
        </div>
    )
}

export default FeedbackBox;