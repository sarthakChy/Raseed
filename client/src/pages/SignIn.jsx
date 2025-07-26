import SignInCard from "../components/SignInCard";
import React from "react";
import Header from "../components/Header";

export default function SignIn() {
  return (
    <>
      <Header />
      <div className="h-full py-12 flex justify-center items-center">
        <SignInCard />
      </div>   
    </>
  );
}
