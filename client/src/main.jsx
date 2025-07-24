// import React from "react";
// import ReactDOM from "react-dom/client";
// import App from "./App";
// import { AuthProvider } from "./context/AuthContext";
// import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// const queryClient = new QueryClient();

// ReactDOM.createRoot(document.getElementById("root")).render(
//   <React.StrictMode>
//     <AuthProvider>
//       <QueryClientProvider client={queryClient}>
//         <App />
//       </QueryClientProvider>
//     </AuthProvider>
//   </React.StrictMode>
// );


import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { AuthProvider } from './context/AuthContext';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Could not find root element to mount to');
}

const root = ReactDOM.createRoot(rootElement);
const queryClient = new QueryClient();

root.render(
  <React.StrictMode>
    <AuthProvider>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </AuthProvider>
  </React.StrictMode>
);
