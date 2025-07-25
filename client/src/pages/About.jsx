import React from "react";
import {
  FaRobot,
  FaCloudUploadAlt,
  FaFileInvoice,
  FaStar,
  FaChevronLeft,
  FaChevronRight,
} from "react-icons/fa";
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
} from "react-icons/si";
import Header from "../components/Header"; // ✅ Added

// ⭐ StarRating Component
const StarRating = ({ rating }) => (
  <div className="flex text-yellow-400 mb-2">
    {[...Array(5)].map((_, i) => (
      <FaStar key={i} color={i < rating ? undefined : "lightgray"} />
    ))}
  </div>
);

// 🏷️ TechTag Component
const TechTag = ({ children, className = "" }) => (
  <span
    className={`px-4 py-2 border rounded-lg text-sm font-medium transition ${className}`}
  >
    {children}
  </span>
);

const About = () => {
  return (
    <>
      <Header /> {/* ✅ Added drawer/mobile header */}

      <div className="bg-white font-sans">
      {/* What's RASEED */}
      <section className="py-16 px-4">
        <div className="container mx-auto flex flex-col md:flex-row items-center justify-center">
          <div className="md:w-1/2 text-center md:text-left">
            <h2 className="text-4xl font-bold text-blue-600 mb-4">
              What's RASEED?
            </h2>
            <p className="text-xl text-gray-700 mb-2">
              Raseed is an AI platform that helps you make sense of your
              receipts and meetings.
            </p>
            <p className="text-gray-600">
              It organizes your data, finds patterns, and reminds you of what to
              do next – so nothing slips through the cracks.
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
            <h2 className="text-4xl font-bold text-red-500 mb-4">
              RASEED is Built to Bring Clarity to Chaos.
            </h2>
            <p className="text-xl text-gray-700 mb-2">
              Organizing receipts and finances shouldn't be a burden.
            </p>
            <p className="text-gray-600">
              RASEED transforms your scattered records into valuable, actionable
              insights.
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
        <h2 className="text-4xl font-bold text-yellow-600 mb-8">
          How will RASEED help?
        </h2>
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

      {/* Features */}
      <section className="py-16 px-4">
        <h2 className="text-3xl font-bold text-center mb-8">
          Features of RASEED
        </h2>
        <div className="relative">
          <div className="border-l-2 border-gray-400 absolute h-full left-1/2 -translate-x-1/2"></div>
          <div className="space-y-8">
            {[
              ["Smart Uploads", "Upload Receipts via Photo, Video, or Stream", "bg-blue-500"],
              ["Multilingual Parsing", "Multilingual Receipt Parsing", "bg-red-500"],
              ["Regional Language Queries", "Ask Queries in Hindi, Tamil, Arabic & More", "bg-yellow-500"],
              ["Insights & Trends", "See Spending Patterns & Trends", "bg-green-500"],
              ["Google Wallet Integration", "Structured Receipt Pass in Google Wallet", "bg-blue-500"],
              ["Reorder Suggestions", "Shopping List & Reorder Recommendations", "bg-red-500"],
            ].map(([title, desc, color], index) => {
              const isLeft = index % 2 === 0;
              return (
                <div className="flex justify-center items-center" key={index}>
                  <div className="w-5/12 text-right pr-8">
                    {isLeft && (
                      <>
                        <h3 className="text-xl font-semibold">{title}</h3>
                        <p>{desc}</p>
                      </>
                    )}
                  </div>
                  <div className="w-1/12 flex justify-center">
                    <div className={`w-4 h-4 ${color} rounded-full z-10`} />
                  </div>
                  <div className="w-5/12 text-left pl-8">
                    {!isLeft && (
                      <>
                        <h3 className="text-xl font-semibold">{title}</h3>
                        <p>{desc}</p>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <hr className="border-dashed container mx-auto" />

      {/* Tech Stack */}
      <section className="py-16 px-4 text-center">
        <h2 className="text-4xl font-bold text-red-500 mb-12">
          How's RASEED built?
        </h2>
        <div className="container mx-auto">
          <h3 className="text-2xl font-semibold mb-4">
            Google Cloud & AI Tools
          </h3>
          <div className="flex justify-center items-center space-x-6 text-5xl text-gray-700 p-4 bg-gray-100 rounded-lg">
            <SiGooglecloud className="text-blue-500" />
            <SiFirebase className="text-yellow-500" />
          </div>

          <h3 className="text-2xl font-semibold mb-4 mt-12">
            Design, Implementation, and Working
          </h3>
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
        <h2 className="text-4xl font-bold text-center mb-12">
          User Testimonials
        </h2>
        <div className="container mx-auto flex items-center justify-center">
          <button className="text-3xl text-gray-400 hover:text-gray-600">
            <FaChevronLeft />
          </button>
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
          <button className="text-3xl text-gray-400 hover:text-gray-600">
            <FaChevronRight />
          </button>
        </div>
      </section>
    </div>

    </>
    
  );
};

export default About;
