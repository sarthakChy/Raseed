import React, { useState, useEffect } from "react";
import HeroIllustration from "../components/HeroIllustration";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header"; // ✅ Mobile drawer header

const Hero = () => {
  const [isVisible, setIsVisible] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  return (
    <>
      <Header />
      <main
        className={`pt-20 pb-12 px-4 sm:px-6 md:pt-24 md:pb-20 transition-opacity duration-1000 ${
          isVisible ? "opacity-100" : "opacity-0"
        }`}
      >
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          <div className="space-y-6 text-center md:text-left">
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-extrabold leading-tight tracking-tight">
              Get AI-powered insights from your meetings & receipts — instantly
            </h1>
            <p className="text-base sm:text-lg md:text-xl text-gray-600 max-w-xl mx-auto md:mx-0">
              Raseed organizes your transactions, extracts patterns, and gives
              follow-ups — all in one place.
            </p>
            <div className="flex flex-col sm:flex-row justify-center md:justify-start space-y-4 sm:space-y-0 sm:space-x-4 pt-4">
              <button
                onClick={() => navigate("/getstarted")}
                className="bg-[#EF4444] text-white font-semibold py-3 px-8 rounded-lg shadow-lg hover:bg-red-600 transition-all duration-300 transform hover:scale-105"
              >
                Get Started
              </button>
              <button className="bg-white text-gray-900 font-semibold py-3 px-8 rounded-lg border-2 border-gray-900 shadow-sm hover:bg-gray-100 transition-all duration-300 transform hover:scale-105">
                Watch Demo
              </button>
            </div>
          </div>
          <div className="relative hidden sm:block">
            <HeroIllustration isVisible={isVisible} />
          </div>
        </div>
      </main>
    </>
  );
};

export default Hero;
