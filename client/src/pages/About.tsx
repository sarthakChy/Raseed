
import React from 'react';

// Helper component for Tech Stack tags
const TechTag = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <button className={`bg-white border border-gray-300 rounded-lg px-6 py-2 text-lg font-medium text-gray-700 hover:bg-gray-100 transition ${className}`}>
    {children}
  </button>
);

// SVG Icon Components
const AIDrivenInsightsIcon = () => (
    <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M30 5L55 50H5L30 5Z" fill="#4285F4"/>
        <circle cx="12" cy="45" r="7" fill="#FBBC05"/>
        <path d="M48 45C48 36.7157 41.2843 30 33 30C24.7157 30 18 36.7157 18 45" stroke="#34A853" strokeWidth="6"/>
    </svg>
);

const TransactionOrganizationIcon = () => (
    <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect y="10" width="45" height="8" rx="4" fill="#34A853"/>
        <rect y="26" width="45" height="8" rx="4" fill="#34A853"/>
        <rect y="42" width="45" height="8" rx="4" fill="#34A853"/>
    </svg>
);

const PatternRecognitionIcon = () => (
    <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M5 50L20 10L35 50L50 10" stroke="#FBBC05" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
);

const FollowUpSuggestionsIcon = () => (
    <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="30" cy="30" r="25" fill="#EA4335"/>
        <path d="M20 30L28 38L42 24" stroke="white" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
);

const ProblemIllustration = () => (
    <svg width="320" height="200" viewBox="0 0 320 200" className="w-full max-w-sm">
        {/* Background Window */}
        <rect x="20" y="10" width="200" height="120" rx="12" fill="white" stroke="#4A5568" strokeWidth="2.5"/>
        <rect x="20" y="10" width="200" height="25" rx="0" fill="#F7FAFC" style={{ borderTopLeftRadius: '12px', borderTopRightRadius: '12px' }}/>
        <circle cx="35" cy="22.5" r="4" fill="#E2E8F0"/>
        <circle cx="50" cy="22.5" r="4" fill="#E2E8F0"/>
        <circle cx="65" cy="22.5" r="4" fill="#E2E8F0"/>
        
        {/* Receipt inside Background Window */}
        <rect x="60" y="50" width="120" height="70" rx="8" fill="#FBBF24"/>
        <rect x="75" y="65" width="90" height="8" rx="2" fill="#4285F4"/>
        <rect x="75" y="80" width="90" height="8" rx="2" fill="#4285F4"/>
        <rect x="75" y="95" width="90" height="8" rx="2" fill="#4285F4"/>
        <path d="M60 110 H180 L170 120 H70Z" fill="#FBBF24"/>

        {/* Foreground Window */}
        <rect x="120" y="60" width="180" height="130" rx="12" fill="white" stroke="#4A5568" strokeWidth="2.5"/>
        <rect x="120" y="60" width="180" height="25" rx="0" fill="#F7FAFC" style={{ borderTopLeftRadius: '12px', borderTopRightRadius: '12px' }}/>
        <circle cx="135" cy="72.5" r="4" fill="#E2E8F0"/>
        <circle cx="150" cy="72.5" r="4" fill="#E2E8F0"/>
        <circle cx="165" cy="72.5" r="4" fill="#E2E8F0"/>
        <rect x="123" y="88" width="174" height="99" rx="10" fill="#34A853"/>
        
        {/* Person Illustration */}
        <path d="M230 180 C 210 180, 200 160, 200 140 C 200 120, 210 100, 230 100 C 250 100, 260 120, 260 140 C 260 160, 250 180, 230 180 Z" fill="#2D3748" />
        <path d="M230 100 Q 235 120 250 115" stroke="white" fill="none" strokeWidth="2"/>
        <circle cx="245" cy="120" r="3" fill="white"/>
        <circle cx="215" cy="120" r="3" fill="white"/>
        <path d="M258 135 a 10 10 0 0 1 0 20" fill="none" stroke="white" strokeWidth="3"/>
        <path d="M260 148 L 265 148 L 265 155" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round"/>
        <circle cx="268" cy="158" r="3" fill="white"/>
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
        </div>
    );
};

export default About;


