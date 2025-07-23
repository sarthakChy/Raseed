
import React from 'react';
import { FaRobot, FaCloudUploadAlt, FaFileInvoice, } from 'react-icons/fa';

import { FaStar, FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import {
    SiGooglecloud,
    SiNextdotjs,
    SiReact,
    SiJavascript,
    SiPython,
    SiNodedotjs,
    SiMongodb,
    SiFirebase,
    SiPostman
} from 'react-icons/si';

// A simple star rating component for testimonials
const StarRating = ({ rating }: { rating: number }) => (
    <div className="flex text-yellow-400 mb-2">
        {[...Array(5)].map((_, i) => (
            <FaStar key={i} color={i < rating ? undefined : 'lightgray'} />
        ))}
    </div>
);

// Individual component for each feature on the timeline
const FeatureItem = ({ side, color, title, description }: { side: 'left' | 'right', color: string, title: string, description: string }) => (
    <div className={`flex justify-between items-center w-full`}>
        {side === 'right' && <div className="w-5/12"></div>}
        <div className="w-1/12 flex justify-center">
            <div className={`w-4 h-4 rounded-full z-10`} style={{ backgroundColor: color }}></div>
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

            {/* Section: What's RASEED? */}
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

            {/* Section: Bring Clarity to Chaos */}
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

            {/* Section: How will RASEED help? */}
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

            {/* Section: How RASEED Works? */}
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

            {/* Section: Features */}
            <section className="py-16 px-4">
                <div className="mb-12">
                    <h2 className="text-3xl font-bold text-center mb-8">Features of RASEED</h2>
                    <div className="relative">
                        <div className="border-l-2 border-gray-400 absolute h-full left-1/2 -translate-x-1/2"></div>
                        <div className="space-y-8">
                            <div className="flex justify-center items-center">
                                <div className="w-5/12 text-right pr-8">
                                    <h3 className="text-xl font-semibold">Smart Uploads</h3>
                                    <p>Upload Receipts via Photo, Video, or Stream</p>
                                </div>
                                <div className="w-1/12 flex justify-center">
                                    <div className="w-4 h-4 bg-blue-500 rounded-full z-10"></div>
                                </div>
                                <div className="w-5/12"></div>
                            </div>
                            <div className="flex justify-center items-center">
                                <div className="w-5/12"></div>
                                <div className="w-1/12 flex justify-center">
                                    <div className="w-4 h-4 bg-red-500 rounded-full z-10"></div>
                                </div>
                                <div className="w-5/12 text-left pl-8">
                                    <h3 className="text-xl font-semibold">Multilingual Parsing</h3>
                                    <p>Multilingual Receipt Parsing</p>
                                </div>
                            </div>
                            <div className="flex justify-center items-center">
                                <div className="w-5/12 text-right pr-8">
                                    <h3 className="text-xl font-semibold">Regional Language Queries</h3>
                                    <p>Ask Queries in Hindi, Tamil, Arabic & More</p>
                                </div>
                                <div className="w-1/12 flex justify-center">
                                    <div className="w-4 h-4 bg-yellow-500 rounded-full z-10"></div>
                                </div>
                                <div className="w-5/12"></div>
                            </div>
                            <div className="flex justify-center items-center">
                                <div className="w-5/12"></div>
                                <div className="w-1/12 flex justify-center">
                                    <div className="w-4 h-4 bg-green-500 rounded-full z-10"></div>
                                </div>
                                <div className="w-5/12 text-left pl-8">
                                    <h3 className="text-xl font-semibold">Insights & Trends</h3>
                                    <p>See Spending Patterns & Trends</p>
                                </div>
                            </div>
                            <div className="flex justify-center items-center">
                                <div className="w-5/12 text-right pr-8">
                                    <h3 className="text-xl font-semibold">Google Wallet Integration</h3>
                                    <p>Structured Receipt Pass in Google Wallet</p>
                                </div>
                                <div className="w-1/12 flex justify-center">
                                    <div className="w-4 h-4 bg-blue-500 rounded-full z-10"></div>
                                </div>
                                <div className="w-5/12"></div>
                            </div>
                            <div className="flex justify-center items-center">
                                <div className="w-5/12"></div>
                                <div className="w-1/12 flex justify-center">
                                    <div className="w-4 h-4 bg-red-500 rounded-full z-10"></div>
                                </div>
                                <div className="w-5/12 text-left pl-8">
                                    <h3 className="text-xl font-semibold">Reorder Suggestions</h3>
                                    <p>Shopping List & Reorder Recommendations</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

            </section>

            <hr className="border-dashed container mx-auto" />

            {/* Section: How it's Built */}
            <section className="py-16 px-4 text-center">
                <h2 className="text-4xl font-bold text-red-500 mb-12">How's RASEED built?</h2>
                <div className="container mx-auto">
                    <h3 className="text-2xl font-semibold mb-4">Google Cloud & AI Tools</h3>
                    <div className="flex justify-center items-center space-x-6 text-5xl text-gray-700 p-4 bg-gray-100 rounded-lg">
                        <SiGooglecloud className="text-blue-500" />
                        <SiFirebase className="text-yellow-500" />
                        {/* Add other Google Cloud icons if available/needed */}
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

            {/* Section: User Testimonials */}
            <section className="py-16 px-4 bg-gray-50">
                <h2 className="text-4xl font-bold text-center mb-12">User Testimonials</h2>
                <div className="container mx-auto flex items-center justify-center">
                    <button className="text-3xl text-gray-400 hover:text-gray-600"><FaChevronLeft /></button>
                    <div className="flex flex-col md:flex-row space-y-8 md:space-y-0 md:space-x-8 px-4">
                        <div className="bg-white p-6 rounded-lg shadow-lg text-center max-w-xs">
                            <StarRating rating={5} />
                            <p className="text-gray-600 italic mb-4">"The way RASEED breaks down insights is perfect for students managing budgets."</p>
                            <h4 className="font-bold">Ayesha Khan</h4>
                            <p className="text-sm text-gray-500">Finance Student</p>
                        </div>
                        <div className="bg-white p-6 rounded-lg shadow-lg text-center max-w-xs">
                            <StarRating rating={5} />
                            <p className="text-gray-600 italic mb-4">"Helped me track 50+ client payments this quarter. The follow-up reminders saved me from chasing emails all week!"</p>
                            <h4 className="font-bold">Neha Kapoor</h4>
                            <p className="text-sm text-gray-500">Freelance Consultant</p>
                        </div>
                        <div className="bg-white p-6 rounded-lg shadow-lg text-center max-w-xs">
                            <StarRating rating={4} />
                            <p className="text-gray-600 italic mb-4">"No more chaos. Everything from Amazon to utility bills—auto-sorted & insightful."</p>
                            <h4 className="font-bold">Mark Thomas</h4>
                            <p className="text-sm text-gray-500">Remote Executive</p>
                        </div>
                    </div>
                    <button className="text-3xl text-gray-400 hover:text-gray-600"><FaChevronRight /></button>
                </div>
            </section>
        </div>
    );
};

export default AboutPage;


