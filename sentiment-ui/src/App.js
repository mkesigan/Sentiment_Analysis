import React, { useState } from "react";
import { Bar } from "react-chartjs-2";
import "chart.js/auto";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeSentiment = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: text }),
      });

      const data = await res.json();
      console.log("API Response:", data);
      setResult(data);

    } catch (error) {
      console.error("Fetch error:", error);
      alert("Could not connect to backend!");
    }

    setLoading(false);
  };

  const chartData =
    result && {
      labels: ["Positive", "Negative"],
      datasets: [
        {
          label: "Confidence",
          data:
            result.sentiment === "positive"
              ? [result.confidence, 1 - result.confidence]
              : [1 - result.confidence, result.confidence],
          backgroundColor: ["#10B981", "#EF4444"],
        },
      ],
    };

  return (
    <div className="min-h-screen bg-gray-50 p-6 flex justify-center items-start">
      <div className="w-full max-w-3xl bg-white rounded-xl shadow-xl p-8">
        <h1 className="text-4xl font-bold text-center text-indigo-700">
          Sentiment Analysis (DistilBERT)
        </h1>

        <p className="text-gray-600 text-center mt-2 mb-6">
          Enter a sentence to analyze its sentiment using your fine-tuned model.
        </p>

        <textarea
          className="w-full border p-4 rounded-lg mb-4 focus:ring-2 focus:ring-indigo-500"
          placeholder="Type something..."
          rows="4"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button
          onClick={analyzeSentiment}
          className="w-full bg-indigo-600 text-white py-3 rounded-lg hover:bg-indigo-700 transition"
        >
          {loading ? "Analyzing..." : "Analyze Sentiment"}
        </button>

        {result && (
          <div className="mt-8">
            <h2 className="text-2xl font-semibold mb-3">🎯 Prediction Result</h2>

            <div className="bg-gray-100 border-l-4 border-indigo-500 p-4 rounded">
              <p className="text-xl">
                Sentiment:{" "}
                <b
                  className={
                    result.sentiment === "positive"
                      ? "text-green-600"
                      : "text-red-600"
                  }
                >
                  {result.sentiment.toUpperCase()}
                </b>
              </p>

              <p className="mt-2 text-gray-700">
                Confidence:{" "}
                <b>{(result.confidence * 100).toFixed(2)}%</b>
              </p>

              <p className="mt-2 text-gray-600">
                Cleaned Text: {result.clean_text}
              </p>
            </div>

            {/* Chart */}
            <div className="mt-6 bg-white p-4 rounded-xl shadow">
              <Bar data={chartData} />
            </div>

            {/* Explanation */}
            <div className="mt-6 bg-gray-100 p-4 rounded-xl">
              <h3 className="text-xl font-semibold mb-2">🧠 Explanation</h3>
              <p className="text-gray-700 leading-relaxed">
                Your input text is tokenized and processed through a fine-tuned
                DistilBERT model. The model outputs probability scores for
                POSITIVE and NEGATIVE categories.
              </p>
              <p className="mt-2 text-gray-700">
                The final prediction is based on the higher confidence score.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
