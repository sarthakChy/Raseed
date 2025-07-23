import React from 'react';

const HeroIllustration = ({ isVisible }) => {
  const baseTransition = 'transition-all duration-1000 ease-out';

  return (
    <div className="relative w-full h-96 min-h-[400px] flex justify-center items-center mt-10 md:mt-0">
      <div className="relative w-[350px] h-[350px] sm:w-[400px] sm:h-[400px]">
        {/* Wallet */}
        <div
          className={`absolute bottom-0 left-1/2 -translate-x-1/2 w-[280px] h-[150px] transform ${baseTransition} ${
            isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
          }`}
          style={{ transitionDelay: '200ms' }}
        >
          <div className="relative w-full h-full">
            <div className="absolute bottom-0 w-full h-[130px] bg-blue-600 rounded-t-2xl rounded-b-lg border-2 border-black"></div>
            <div className="absolute bottom-0 w-[95%] left-[2.5%] h-[140px] bg-blue-500 rounded-t-xl"></div>
            <div className="absolute -top-[18px] w-full h-10 bg-blue-600 rounded-t-xl border-2 border-b-0 border-black"></div>
            <div className="absolute bottom-4 right-4 w-7 h-7 bg-yellow-400 rounded-full border-2 border-black"></div>
          </div>
        </div>

        {/* Receipt */}
        <div
          className={`absolute bottom-[100px] left-1/2 -translate-x-[45%] w-[180px] h-[160px] transform ${baseTransition} ${
            isVisible ? 'translate-y-0 opacity-100' : 'translate-y-16 opacity-0'
          }`}
          style={{ transitionDelay: '400ms' }}
        >
          <div className="w-full h-full bg-white border-2 border-black rounded-t-lg p-4 pt-6 space-y-3 relative">
            <div className="w-full h-3 bg-blue-300 rounded"></div>
            <div className="w-full h-3 bg-blue-300 rounded"></div>
            <div className="w-3/4 h-3 bg-blue-300 rounded"></div>
            <div className="w-full h-3 bg-blue-300 rounded"></div>
            <div className="absolute -bottom-5 right-[-30px] text-red-500 text-3xl font-bold">$</div>
          </div>
          <svg className="absolute top-[-1px] left-0 w-full text-black" viewBox="0 0 100 10" preserveAspectRatio="none" fill="white">
            <path d="M0 10 Q 5 0, 10 10 T 20 10 T 30 10 T 40 10 T 50 10 T 60 10 T 70 10 T 80 10 T 90 10 T 100 10" stroke="black" strokeWidth="2" fill="none" />
          </svg>
        </div>

        {/* Coin */}
        <div
          className={`absolute bottom-[150px] left-[10px] w-20 h-20 bg-yellow-400 rounded-full flex items-center justify-center border-2 border-black shadow-xl transform animate-[float_3s_ease-in-out_infinite] ${baseTransition} ${
            isVisible ? 'scale-100 opacity-100' : 'scale-0 opacity-0'
          }`}
          style={{ animationDelay: '-1.5s', transitionDelay: '600ms' }}
        >
          <span className="text-black text-4xl font-bold">$</span>
        </div>

        {/* Question Bubble */}
        <div
          className={`absolute top-0 left-1/2 -translate-x-[40%] w-[120px] h-[90px] transform animate-[float_3s_ease-in-out_infinite] ${baseTransition} ${
            isVisible ? 'scale-100 opacity-100' : 'scale-0 opacity-0'
          }`}
          style={{ transitionDelay: '700ms' }}
        >
          <div className="relative w-full h-full">
            <div className="w-full h-full bg-green-500 rounded-2xl flex items-center justify-center border-2 border-black shadow-xl">
              <span className="text-white text-5xl font-extrabold">?</span>
            </div>
            <div
              className="absolute -bottom-[15px] left-8 w-0 h-0"
              style={{
                borderLeft: '15px solid transparent',
                borderRight: '15px solid transparent',
                borderTop: '20px solid black',
              }}
            ></div>
            <div
              className="absolute -bottom-[12.5px] left-8 w-0 h-0"
              style={{
                borderLeft: '15px solid transparent',
                borderRight: '15px solid transparent',
                borderTop: '20px solid #22c55e',
              }}
            ></div>
          </div>
        </div>

        {/* Small Bar Chart */}
        <div
          className={`absolute top-[40px] right-0 w-[120px] h-[70px] flex items-end justify-center space-x-2 p-1 transform animate-[float_3s_ease-in-out_infinite] ${baseTransition} ${
            isVisible ? 'scale-100 opacity-100' : 'scale-0 opacity-0'
          }`}
          style={{ animationDelay: '-0.5s', transitionDelay: '800ms' }}
        >
          <div className="w-1/3 h-1/2 bg-red-400 border-2 border-black rounded-t-sm"></div>
          <div className="w-1/3 h-3/4 bg-blue-400 border-2 border-black rounded-t-sm"></div>
          <div className="w-1/3 h-full bg-yellow-400 border-2 border-black rounded-t-sm"></div>
        </div>

        {/* Main Chart */}
        <div
          className={`absolute bottom-[110px] right-[-20px] w-[200px] h-[130px] bg-white rounded-lg border-2 border-black shadow-2xl p-3 transform ${baseTransition} ${
            isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
          }`}
          style={{ transitionDelay: '500ms' }}
        >
          <svg className="w-full h-[60%]" viewBox="0 0 100 50">
            <path d="M 10 40 C 30 10, 50 20, 70 15 L 90 5" stroke="black" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M 85 2 L 90 5 L 85 8" stroke="black" strokeWidth="2" fill="none" />
          </svg>
          <div className="absolute bottom-2 left-0 right-0 flex items-end justify-evenly h-10 px-4">
            <div className="w-5 h-4 bg-red-400 border-2 border-black rounded-sm"></div>
            <div className="w-5 h-8 bg-blue-400 border-2 border-black rounded-sm"></div>
            <div className="w-5 h-6 bg-yellow-400 border-2 border-black rounded-sm"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeroIllustration;
