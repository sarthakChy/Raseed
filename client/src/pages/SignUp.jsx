import SignUpCard from "../components/SignUpCard";
import React from "react";
import Header from "../components/Header";

export default function SignUp() {
  return (
    <>
      <Header />
      <div className="h-full flex py-12 justify-center items-center">
        <SignUpCard />
      </div>  
    </>
  );
}
