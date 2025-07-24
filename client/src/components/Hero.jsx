import React, { useState, useEffect } from 'react';
import HeroIllustration from './HeroIllustration';
import { useNavigate } from 'react-router-dom';

const Hero = () => {

  const navigate = useNavigate();
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  return (
    <main className={`py-12 md:py-20 transition-opacity duration-1000 ${isVisible ? 'opacity-100' : 'opacity-0'}`}>
      <div className="grid md:grid-cols-2 gap-16 items-center">
        <div className="space-y-6 text-center md:text-left">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold leading-tight tracking-tighter">
            Get AI-powered insights from your meetings & receipts — instantly
          </h1>
          <p className="text-lg md:text-xl text-gray-600 max-w-lg mx-auto md:mx-0">
            Raseed organizes your transactions, extracts patterns, and gives follow-ups — all in one place.
          </p>
          <div className="flex flex-col sm:flex-row justify-center md:justify-start space-y-4 sm:space-y-0 sm:space-x-4 pt-4">
            <button
<<<<<<< HEAD

              onClick={() => navigate('/getstarted')}
              className="bg-[#EF4444] text-white font-semibold py-3 px-8 rounded-lg shadow-lg hover:bg-red-600 transition-all duration-300 transform hover:scale-105"
=======
<<<<<<< HEAD
<<<<<<< HEAD
              onClick={() => navigate('/getstarted')}
              className="bg-[#EF4444] text-white font-semibold py-3 px-8 rounded-lg shadow-lg hover:bg-red-600 transition-all duration-300 transform hover:scale-105"
=======
              className="bg-[#EF4444] text-white font-semibold py-3 px-8 rounded-lg shadow-lg hover:bg-red-600 transition-all duration-300 transform hover:scale-105"
              onClick={() => navigate('/GetStarted')}
>>>>>>> 1cd74f5 (TSX -> JSX + Auth-Setup)
=======
<<<<<<< HEAD:client/src/components/Hero.jsx
              className="bg-[#EF4444] text-white font-semibold py-3 px-8 rounded-lg shadow-lg hover:bg-red-600 transition-all duration-300 transform hover:scale-105"
              onClick={() => navigate('/GetStarted')}
=======
              onClick={() => navigate('/getstarted')}
              className="bg-[#EF4444] text-white font-semibold py-3 px-8 rounded-lg shadow-lg hover:bg-red-600 transition-all duration-300 transform hover:scale-105"
>>>>>>> fb50a81 ( change in about and home):client/src/components/Hero.tsx
>>>>>>> c222fd6 ( change in about and home)
>>>>>>> 00a20c1 (I have no idea what is this at this point)
            >
              Get Started
            </button>
            <button className="bg-white text-gray-900 font-semibold py-3 px-8 rounded-lg border-2 border-gray-900 shadow-sm hover:bg-gray-100 transition-all duration-300 transform hover:scale-105">
              Watch Demo
            </button>
          </div>
        </div>
        <div className="relative">
          <HeroIllustration isVisible={isVisible} />
        </div>
      </div>
    </main>
  );
};

export default Hero;
