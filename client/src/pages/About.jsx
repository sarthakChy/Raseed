import React from 'react';
<<<<<<< HEAD
import { FaRobot, FaCloudUploadAlt, FaFileInvoice, FaStar, FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import {
  SiGooglecloud,
  SiNextdotjs,
  SiReact,
  SiJavascript,
  SiPython,
  SiNodedotjs,
  SiMongodb,
  SiFirebase,
  SiPostman,
} from 'react-icons/si';

// StarRating component
const StarRating = ({ rating }) => (
  <div className="flex text-yellow-400 mb-2">
    {[...Array(5)].map((_, i) => (
      <FaStar key={i} color={i < rating ? undefined : 'lightgray'} />
    ))}
  </div>
);

// FeatureItem component (no longer used but left for reference)
const FeatureItem = ({ side, color, title, description }) => (
  <div className="flex justify-between items-center w-full">
    {side === 'right' && <div className="w-5/12"></div>}
    <div className="w-1/12 flex justify-center">
      <div className="w-4 h-4 rounded-full z-10" style={{ backgroundColor: color }}></div>
    </div>
    <div className={`w-5/12 px-4 ${side === 'left' ? 'text-right' : 'text-left'}`}>
      <h3 className="font-bold text-lg">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
    {side === 'left' && <div className="w-5/12"></div>}
  </div>
);

const AboutPage = () => {
  return (
    <div className="bg-white font-sans">
      {/* What's RASEED */}
      <section className="py-16 px-4">
        <div className="container mx-auto flex flex-col md:flex-row items-center justify-center">
          <div className="md:w-1/2 text-center md:text-left">
            <h2 className="text-4xl font-bold text-blue-600 mb-4">What's RASEED?</h2>
            <p className="text-xl text-gray-700 mb-2">
              Raseed is an AI platform that helps you make sense of your receipts and meetings.
            </p>
            <p className="text-gray-600">
              It organizes your data, finds patterns, and reminds you of what to do next – so nothing slips through the cracks.
            </p>
          </div>
          <div className="md:w-1/2 flex justify-center mt-8 md:mt-0">
            <div className="w-64 h-64 rounded-full bg-blue-50 overflow-hidden flex items-end">
              <div className="w-full h-1/2 bg-green-200 rounded-t-full"></div>
            </div>
          </div>
        </div>
      </section>

      <hr className="border-dashed container mx-auto" />

      {/* Bring Clarity to Chaos */}
      <section className="py-16 px-4">
        <div className="container mx-auto flex flex-col md:flex-row-reverse items-center justify-center">
          <div className="md:w-1/2 text-center md:text-left md:pl-12">
            <h2 className="text-4xl font-bold text-red-500 mb-4">RASEED is Built to Bring Clarity to Chaos.</h2>
            <p className="text-xl text-gray-700 mb-2">
              Organizing receipts and finances shouldn't be a burden.
            </p>
            <p className="text-gray-600">
              RASEED transforms your scattered records into valuable, actionable insights.
            </p>
          </div>
          <div className="md:w-1/2 flex justify-center mt-8 md:mt-0">
            <div className="w-48 h-48 rounded-full bg-blue-50 overflow-hidden flex items-end">
              <div className="w-full h-1/2 bg-green-200 rounded-t-full"></div>
            </div>
          </div>
        </div>
      </section>

      <hr className="border-dashed container mx-auto" />

      {/* How will RASEED help? */}
      <section className="py-16 px-4 text-center">
        <h2 className="text-4xl font-bold text-yellow-600 mb-8">How will RASEED help?</h2>
        <div className="container mx-auto flex flex-col md:flex-row items-center justify-around">
          <div className="text-gray-700 text-xl italic space-y-4">
            <p>"Receipts & meeting notes are messy and often ignored."</p>
            <p>"We help you extract meaning from them – instantly."</p>
            <p>"No more manual follow-ups or forgotten expenses."</p>
          </div>
          <div className="mt-8 md:mt-0">
            <div className="w-48 h-48 rounded-full border-2 border-dashed border-gray-400 flex items-center justify-center p-4">
              <p className="text-gray-600">Receipts + Mess → AI → Insights</p>
            </div>
          </div>
        </div>
      </section>

      <hr className="border-dashed container mx-auto" />

      {/* How RASEED Works */}
      <section className="py-16 px-4 text-center">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-8">How RASEED Works?</h2>
          <div className="flex flex-col md:flex-row justify-center items-center space-y-8 md:space-y-0 md:space-x-8">
            <div className="flex flex-col items-center">
              <FaCloudUploadAlt className="text-5xl text-red-500 mb-2" />
              <h3 className="text-xl font-semibold">Upload / Sync</h3>
              <p>Connect Gmail or Scan Receipts Directly</p>
            </div>
            <div className="text-5xl text-gray-400">→</div>
            <div className="flex flex-col items-center">
              <FaRobot className="text-5xl text-blue-500 mb-2" />
              <h3 className="text-xl font-semibold">AI Analysis</h3>
              <p>Patterns detected, summaries generated</p>
            </div>
            <div className="text-5xl text-gray-400">→</div>
            <div className="flex flex-col items-center">
              <FaFileInvoice className="text-5xl text-yellow-500 mb-2" />
              <h3 className="text-xl font-semibold">Insights & Follow-Ups</h3>
              <p>Get recommendations, alerts, & charts</p>
            </div>
          </div>
        </div>
      </section>

      <hr className="border-dashed container mx-auto" />

      {/* Features Section */}
      <section className="py-16 px-4">
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-center mb-8">Features of RASEED</h2>
          <div className="relative">
            <div className="border-l-2 border-gray-400 absolute h-full left-1/2 -translate-x-1/2"></div>
            <div className="space-y-8">
              {[
                ['Smart Uploads', 'Upload Receipts via Photo, Video, or Stream', 'bg-blue-500'],
                ['Multilingual Parsing', 'Multilingual Receipt Parsing', 'bg-red-500'],
                ['Regional Language Queries', 'Ask Queries in Hindi, Tamil, Arabic & More', 'bg-yellow-500'],
                ['Insights & Trends', 'See Spending Patterns & Trends', 'bg-green-500'],
                ['Google Wallet Integration', 'Structured Receipt Pass in Google Wallet', 'bg-blue-500'],
                ['Reorder Suggestions', 'Shopping List & Reorder Recommendations', 'bg-red-500'],
              ].map(([title, desc, color], index) => {
                const isLeft = index % 2 === 0;
                return (
                  <div className="flex justify-center items-center" key={index}>
                    <div className="w-5/12 text-right pr-8">{isLeft && <><h3 className="text-xl font-semibold">{title}</h3><p>{desc}</p></>}</div>
                    <div className="w-1/12 flex justify-center">
                      <div className={`w-4 h-4 ${color} rounded-full z-10`}></div>
                    </div>
                    <div className="w-5/12 text-left pl-8">{!isLeft && <><h3 className="text-xl font-semibold">{title}</h3><p>{desc}</p></>}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      <hr className="border-dashed container mx-auto" />

      {/* How It's Built */}
      <section className="py-16 px-4 text-center">
        <h2 className="text-4xl font-bold text-red-500 mb-12">How's RASEED built?</h2>
        <div className="container mx-auto">
          <h3 className="text-2xl font-semibold mb-4">Google Cloud & AI Tools</h3>
          <div className="flex justify-center items-center space-x-6 text-5xl text-gray-700 p-4 bg-gray-100 rounded-lg">
            <SiGooglecloud className="text-blue-500" />
            <SiFirebase className="text-yellow-500" />
          </div>

          <h3 className="text-2xl font-semibold mb-4 mt-12">Design, Implementation, and Working</h3>
          <div className="flex justify-center items-center flex-wrap gap-6 text-5xl text-gray-700 p-4 bg-gray-100 rounded-lg">
            <SiReact className="text-blue-400" />
            <SiJavascript className="text-yellow-400" />
            <SiPython className="text-blue-600" />
            <SiMongodb className="text-green-500" />
            <SiNodedotjs className="text-green-400" />
            <SiNextdotjs />
            <SiPostman className="text-orange-500" />
          </div>
        </div>
      </section>

      <hr className="border-dashed container mx-auto" />

      {/* User Testimonials */}
      <section className="py-16 px-4 bg-gray-50">
        <h2 className="text-4xl font-bold text-center mb-12">User Testimonials</h2>
        <div className="container mx-auto flex items-center justify-center">
          <button className="text-3xl text-gray-400 hover:text-gray-600"><FaChevronLeft /></button>
          <div className="flex flex-col md:flex-row space-y-8 md:space-y-0 md:space-x-8 px-4">
            {[
              ["The way RASEED breaks down insights is perfect for students managing budgets.", "Ayesha Khan", "Finance Student", 5],
              ["Helped me track 50+ client payments this quarter. The follow-up reminders saved me from chasing emails all week!", "Neha Kapoor", "Freelance Consultant", 5],
              ["No more chaos. Everything from Amazon to utility bills—auto-sorted & insightful.", "Mark Thomas", "Remote Executive", 4],
            ].map(([text, name, role, rating], i) => (
              <div key={i} className="bg-white p-6 rounded-lg shadow-lg text-center max-w-xs">
                <StarRating rating={rating} />
                <p className="text-gray-600 italic mb-4">"{text}"</p>
                <h4 className="font-bold">{name}</h4>
                <p className="text-sm text-gray-500">{role}</p>
              </div>
            ))}
          </div>
          <button className="text-3xl text-gray-400 hover:text-gray-600"><FaChevronRight /></button>
        </div>
      </section>
=======

// Helper component for Tech Stack tags
const TechTag = ({ children, className = '' }) => (
  <button className={`bg-white border border-gray-300 rounded-lg px-6 py-2 text-lg font-medium text-gray-700 hover:bg-gray-100 transition ${className}`}>
    {children}
  </button>
);

// SVG Icon Components
const AIDrivenInsightsIcon = () => (
  <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M30 5L55 50H5L30 5Z" fill="#4285F4" />
    <circle cx="12" cy="45" r="7" fill="#FBBC05" />
    <path d="M48 45C48 36.7157 41.2843 30 33 30C24.7157 30 18 36.7157 18 45" stroke="#34A853" strokeWidth="6" />
  </svg>
);

const TransactionOrganizationIcon = () => (
  <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect y="10" width="45" height="8" rx="4" fill="#34A853" />
    <rect y="26" width="45" height="8" rx="4" fill="#34A853" />
    <rect y="42" width="45" height="8" rx="4" fill="#34A853" />
  </svg>
);

const PatternRecognitionIcon = () => (
  <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M5 50L20 10L35 50L50 10" stroke="#FBBC05" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const FollowUpSuggestionsIcon = () => (
  <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="30" cy="30" r="25" fill="#EA4335" />
    <path d="M20 30L28 38L42 24" stroke="white" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const ProblemIllustration = () => (
  <svg width="320" height="200" viewBox="0 0 320 200" className="w-full max-w-sm">
    <rect x="20" y="10" width="200" height="120" rx="12" fill="white" stroke="#4A5568" strokeWidth="2.5" />
    <rect x="20" y="10" width="200" height="25" fill="#F7FAFC" style={{ borderTopLeftRadius: '12px', borderTopRightRadius: '12px' }} />
    <circle cx="35" cy="22.5" r="4" fill="#E2E8F0" />
    <circle cx="50" cy="22.5" r="4" fill="#E2E8F0" />
    <circle cx="65" cy="22.5" r="4" fill="#E2E8F0" />
    <rect x="60" y="50" width="120" height="70" rx="8" fill="#FBBF24" />
    <rect x="75" y="65" width="90" height="8" rx="2" fill="#4285F4" />
    <rect x="75" y="80" width="90" height="8" rx="2" fill="#4285F4" />
    <rect x="75" y="95" width="90" height="8" rx="2" fill="#4285F4" />
    <path d="M60 110 H180 L170 120 H70Z" fill="#FBBF24" />
    <rect x="120" y="60" width="180" height="130" rx="12" fill="white" stroke="#4A5568" strokeWidth="2.5" />
    <rect x="120" y="60" width="180" height="25" fill="#F7FAFC" style={{ borderTopLeftRadius: '12px', borderTopRightRadius: '12px' }} />
    <circle cx="135" cy="72.5" r="4" fill="#E2E8F0" />
    <circle cx="150" cy="72.5" r="4" fill="#E2E8F0" />
    <circle cx="165" cy="72.5" r="4" fill="#E2E8F0" />
    <rect x="123" y="88" width="174" height="99" rx="10" fill="#34A853" />
    <path d="M230 180 C 210 180, 200 160, 200 140 C 200 120, 210 100, 230 100 C 250 100, 260 120, 260 140 C 260 160, 250 180, 230 180 Z" fill="#2D3748" />
    <path d="M230 100 Q 235 120 250 115" stroke="white" fill="none" strokeWidth="2" />
    <circle cx="245" cy="120" r="3" fill="white" />
    <circle cx="215" cy="120" r="3" fill="white" />
    <path d="M258 135 a 10 10 0 0 1 0 20" fill="none" stroke="white" strokeWidth="3" />
    <path d="M260 148 L 265 148 L 265 155" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" />
    <circle cx="268" cy="158" r="3" fill="white" />
  </svg>
);

// The Main Component
const About = () => {
  const features = [
    {
      icon: <AIDrivenInsightsIcon />,
      title: 'AI-Driven Insights',
      description: 'Leverage advanced AI to gain instant insights from meetings and receipts.',
      titleColor: 'text-blue-600'
    },
    {
      icon: <TransactionOrganizationIcon />,
      title: 'Transaction Organization',
      description: 'Automatically categorize and organize all your transactions.',
      titleColor: 'text-green-600'
    },
    {
      icon: <PatternRecognitionIcon />,
      title: 'Pattern Recognition',
      description: 'Extract patterns and trends from your data effortlessly.',
      titleColor: 'text-yellow-500'
    },
    {
      icon: <FollowUpSuggestionsIcon />,
      title: 'Follow-Up Suggestions',
      description: 'Receive automatic follow-up tasks and reminders.',
      titleColor: 'text-red-500'
    }
  ];

  return (
    <div className="bg-white font-sans text-gray-800">
      <div className="max-w-5xl mx-auto px-6 py-16 space-y-16">

        {/* Section 1: What problem does it solve? */}
        <section className="flex flex-col md:flex-row items-center justify-between gap-12">
          <div className="md:w-1/2 space-y-4">
            <h2 className="text-4xl font-bold">What problem does it solve?</h2>
            <p className="text-lg text-gray-600 leading-relaxed">
              Managing meeting notes and receipts can be time-consuming. Users often face challenges in organizing and extracting meaningful information from these documents, leading to missed insights and opportunities.
            </p>
          </div>
          <div className="md:w-1/2 flex justify-center md:justify-end">
            <ProblemIllustration />
          </div>
        </section>

        <hr className="border-gray-200" />

        {/* Section 2: How it works */}
        <section className="max-w-4xl">
          <h2 className="text-4xl font-bold">How it works</h2>
          <p className="mt-4 text-lg text-gray-600 leading-relaxed">
            Raseed uses artificial intelligence to process meeting summaries and receipts. It identifies key transactions, analyzes for patterns, and provides actionable insights and follow-ups to users quickly and efficiently.
          </p>
        </section>

        <hr className="border-gray-200" />

        {/* Section 3: Core Features */}
        <section>
          <h2 className="text-4xl font-bold">Core Features</h2>
          <div className="mt-10 grid md:grid-cols-2 gap-x-16 gap-y-12">
            {features.map((feature, index) => (
              <div key={index} className="flex items-start space-x-5">
                <div className="flex-shrink-0">{feature.icon}</div>
                <div>
                  <h3 className={`text-xl font-bold ${feature.titleColor}`}>{feature.title}</h3>
                  <p className="mt-1 text-gray-600">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        <hr className="border-gray-200" />

        {/* Section 4: Tech Stack */}
        <section>
          <h2 className="text-4xl font-bold">Tech Stack</h2>
          <div className="mt-8 flex flex-wrap gap-4 items-center">
            <TechTag className="text-blue-500 border-blue-200 hover:bg-blue-50">React</TechTag>
            <TechTag className="text-green-600 border-green-200 hover:bg-green-50">Node.js</TechTag>
            <TechTag className="text-gray-700 border-gray-300">Python</TechTag>
            <TechTag className="text-orange-500 border-orange-200 hover:bg-orange-50">TensorFlow</TechTag>
            <button className="h-12 w-12 flex items-center justify-center bg-white border border-gray-300 rounded-lg text-2xl font-light text-gray-600 hover:bg-gray-100 transition">+</button>
          </div>
        </section>

      </div>
>>>>>>> 1cd74f5 (TSX -> JSX + Auth-Setup)
    </div>
  );
};

<<<<<<< HEAD
export default AboutPage;
=======
export default About;
>>>>>>> 1cd74f5 (TSX -> JSX + Auth-Setup)
