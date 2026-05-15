import React, { useState } from "react";
import { Loader2, Sparkles } from "lucide-react";

export default function NERDemoPage() {
  const [input, setInput] = useState(
    "Barack Obama visited Paris and met with Microsoft executives on Monday."
  );
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const API_URL= import.meta.env.VITE_API_URL 


 

  const handleGenerate = async () => {
    setError("");
    setOutput(null);

    if (!input.trim()) {
      setError("Please enter some text before generating.");
      return;
    }

    try {
      setLoading(true);

      const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: input,
      }),
    });

      const data = await response.json();

      setOutput(data);
    } catch (err) {
      setError("Something went wrong while connecting to the model.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const renderEntities = () => {
    if (!output || !Array.isArray(output)) return null;

    return output.map((entity, index) => (
      <div
        key={index}
        className="bg-white/10 border border-white/10 rounded-xl p-4 mb-3 backdrop-blur-sm"
      >
        <div className="flex justify-between items-center">
          <span className="text-cyan-300 font-semibold text-sm">
            {entity.entity_group || entity.entity}
          </span>

          <span className="text-xs text-gray-400">
            Confidence: {(entity.score * 100).toFixed(2)}%
          </span>
        </div>

        <p className="text-lg text-white mt-2">{entity.word}</p>
      </div>
    ));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-slate-900 to-gray-950 text-white flex items-center justify-center px-6 py-12">
      <div className="w-full max-w-5xl">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-400/20 text-cyan-300 mb-5">
            {/* <Sparkles size={16} /> */}
            Named Entity Recognition Demo
          </div>

          <h1 className="text-5xl font-bold leading-tight">
            AI Powered
            <span className="text-cyan-400"> Entity Recognition</span>
          </h1>

          <p className="text-gray-400 mt-5 text-lg max-w-2xl mx-auto">
            Try out the model by entering any sentence. The AI will identify
            names, locations, organizations, and other entities in real time.
          </p>
        </div>

        {/* Main Card */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Input Section */}
          <div className="bg-white/5 border border-white/10 rounded-3xl p-6 backdrop-blur-xl shadow-2xl">
            <h2 className="text-2xl font-semibold mb-4">
              Input Text
            </h2>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Enter text here..."
              className="w-full h-72 bg-black/30 border border-white/10 rounded-2xl p-5 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-400 resize-none text-base"
            />

            {error && (
              <div className="mt-4 text-red-400 text-sm">
                {error}
              </div>
            )}

            <button
              onClick={handleGenerate}
              disabled={loading}
              className="w-full mt-6 bg-cyan-500 hover:bg-cyan-400 transition-all duration-300 text-black font-semibold py-4 rounded-2xl flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20"
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" size={18} />
                  Generating...
                </>
              ) : (
                "Generate Output"
              )}
            </button>
          </div>

          {/* Output Section */}
          <div className="bg-white/5 border border-white/10 rounded-3xl p-6 backdrop-blur-xl shadow-2xl">
            <h2 className="text-2xl font-semibold mb-4">
              Detected Entities
            </h2>

            <div className="h-72 overflow-y-auto pr-2">
              {!output && !loading && (
                <div className="h-full flex items-center justify-center text-gray-500 text-center">
                  Your detected entities will appear here.
                </div>
              )}

              {loading && (
                <div className="h-full flex flex-col items-center justify-center text-cyan-300">
                  <Loader2 className="animate-spin mb-3" size={30} />
                  Processing your request...
                </div>
              )}

              {renderEntities()}
            </div>

            {output && (
              <div className="mt-6 bg-black/30 rounded-2xl p-4 border border-white/10">
                <p className="text-sm text-gray-400 mb-2">
                  Raw Model Response
                </p>

                <pre className="text-xs text-cyan-200 overflow-auto">
                  {JSON.stringify(output, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        {/* <div className="text-center mt-10 text-gray-500 text-sm">
          Built using React + Hugging Face Inference API
        </div> */}
      </div>
    </div>
  );
}