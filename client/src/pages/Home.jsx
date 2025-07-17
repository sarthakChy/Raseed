import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Spline from "@splinetool/react-spline";
import BlurText from "../components/BlurText";

export default function Home() {
  const [loaded, setLoaded] = useState(false);
  const navigate = useNavigate();

  const handleAnimationComplete = () => {
    console.log("Animation completed!");
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-black">
      {/* Page Title */}
      <title>RASEED — AI Receipts Assistant</title>

      {/* Loading Overlay */}
      {!loaded && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black">
          <p className="text-white text-xl animate-pulse">Loading 3D...</p>
        </div>
      )}

      {/* Spline Background */}
      <Spline
        scene="https://prod.spline.design/94oDWKaxg6oTMPqL/scene.splinecode"
        onLoad={() => setLoaded(true)}
        className="absolute inset-0 w-full h-full z-0 pointer-events-auto"
      />

      {/* Foreground Content */}
      {loaded && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center space-y-6 px-4 z-10 pointer-events-none">
          <BlurText
            text="RASEED"
            delay={50}
            animateBy="letters"
            direction="top"
            onAnimationComplete={handleAnimationComplete}
            className="text-white text-4xl md:text-6xl font-bold pointer-events-auto"
          />

          <p
  className="text-lg md:text-xl font-medium tracking-wide bg-clip-text text-transparent pointer-events-none"
  style={{
    backgroundImage:
      "linear-gradient(90deg, #00ffd0, #007cf0, #00ffd0)",
    backgroundSize: "300% 300%",
    animation: "gradientPulse 6s ease-in-out infinite",
    textShadow: "0 0 8px rgba(0,255,200,0.4), 0 0 20px rgba(0,124,240,0.3)",
  }}
>
  AI-powered Receipts. Smart. Seamless. Secure.
</p>


          <button
            onClick={() => navigate("/signin")}
            className="mt-6 bg-blue-600 hover:bg-blue-500 px-6 py-3 rounded-xl text-white text-lg transition duration-200 pointer-events-auto"
          >
            Try RASEED
          </button>
        </div>
      )}

      {/* Inline CSS animation */}
      <style>
  {`
    @keyframes gradientPulse {
      0%, 100% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
    }
  `}
</style>
    </div>
  );
}
