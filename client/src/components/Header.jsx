import React, { useEffect, useState, useRef } from 'react';
import { useAuth } from "../context/AuthContext";
import { useNavigate, Link } from 'react-router-dom';

const Header = () => {
  const logo = 'raseed-logo.png';
  const dummyAvatar = 'https://i.pravatar.cc/300'; // placeholder avatar
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    setIsAuthenticated(!!user);
  }, [user]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const NavLink = ({ to, children }) => (
    <Link
      to={to}
      className="text-gray-700 hover:text-black transition-colors duration-300 text-base font-medium"
    >
      {children}
    </Link>
  );

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
          <NavLink to="/">Home</NavLink>
          <NavLink to="/About">About</NavLink>
          <NavLink to="/GetStarted">Get Started</NavLink>
          <NavLink to="/Upgrade">Upgrade Plan</NavLink>
          <NavLink to="/faq">FAQs</NavLink>
          <NavLink to="/Contact">Contact</NavLink>

          {isAuthenticated && (
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setDropdownOpen((prev) => !prev)}
                className="w-10 h-10 rounded-full overflow-hidden border-2 border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400"
              >
                <img src={dummyAvatar} alt="User Avatar" className="w-full h-full object-cover" />
              </button>

              {dropdownOpen && (
                <div className="absolute right-0 mt-2 w-40 bg-white border border-gray-200 rounded-lg shadow-lg py-2 z-50">
                  <button
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={() => {
                      setDropdownOpen(false);
                      alert("Option 1 clicked");
                    }}
                  >
                    Option 1
                  </button>
                  <button
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={() => {
                      setDropdownOpen(false);
                      alert("Option 2 clicked");
                    }}
                  >
                    Option 2
                  </button>
                  <button
                    onClick={() => {
                      logout();
                      navigate('/');
                    }}
                    className="block w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-600 hover:text-white ease-in-out"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Mobile Menu Icon */}
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
