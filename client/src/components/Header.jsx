import React from 'react';
import { useAuth } from "../context/AuthContext";
import { useNavigate } from 'react-router-dom';

const NavLink = ({ href, children }) => (
  <a href={href} className="text-gray-700 hover:text-black transition-colors duration-300 text-base font-medium">
    {children}
  </a>
);

const Header = () => {
  const logo = 'raseed-logo.png';
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  return (
    <header className="py-6">
      <nav className="flex justify-between items-center">
        <div className="text-2xl font-extrabold tracking-tight">
          <div className="flex items-center space-x-2">
            <img src={logo} alt="Raseed Logo" className="w-8 h-8 object-contain" />
            <span className="text-2xl font-bold text-gray-800">RASEED</span>
          </div>
        </div>

        <div className="hidden md:flex items-center space-x-8">
          <NavLink href="/">Home</NavLink>
          <NavLink href="/About">About</NavLink>
          <NavLink href="/GetStarted">Get Started</NavLink>
          <NavLink href="/Upgrade">Upgrade Plan</NavLink>
          <NavLink href="/faq">FAQs</NavLink>
          <NavLink href="/Contact">Contact</NavLink>
          {user && (
            <button
              onClick={() => {
                logout();
                navigate('/');
              }}
              className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition"
            >
              Logout
            </button>
          )}
        </div>

        <div className="md:hidden">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
          </svg>
        </div>
      </nav>
    </header>
  );
};

export default Header;
