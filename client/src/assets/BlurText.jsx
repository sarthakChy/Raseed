// import { motion } from "framer-motion";
// import { useEffect, useRef, useState, useMemo } from "react";

// const buildKeyframes = (from, steps) => {
//   const keys = new Set([
//     ...Object.keys(from),
//     ...steps.flatMap((s) => Object.keys(s)),
//   ]);

//   const keyframes = {};
//   keys.forEach((k) => {
//     keyframes[k] = [from[k], ...steps.map((s) => s[k])];
//   });
//   return keyframes;
// };

// const BlurText = ({
//   text = "",
//   delay = 120,
//   className = "",
//   animateBy = "letters",
//   direction = "top",
//   threshold = 0.05,
//   rootMargin = "0px",
//   animationFrom,
//   animationTo,
//   easing = [0.25, 0.1, 0.25, 1], // smoother bezier easing
//   onAnimationComplete,
//   stepDuration = 0.4,
// }) => {
//   const elements = animateBy === "words" ? text.split(" ") : text.split("");
//   const [inView, setInView] = useState(false);
//   const ref = useRef(null);

//   useEffect(() => {
//     if (!ref.current) return;
//     const observer = new IntersectionObserver(
//       ([entry]) => {
//         if (entry.isIntersecting) {
//           setInView(true);
//           observer.unobserve(ref.current);
//         }
//       },
//       { threshold, rootMargin }
//     );
//     observer.observe(ref.current);
//     return () => observer.disconnect();
//   }, [threshold, rootMargin]);

//   const defaultFrom = useMemo(
//     () =>
//       direction === "top"
//         ? { filter: "blur(16px)", opacity: 0, y: -40 }
//         : direction === "bottom"
//         ? { filter: "blur(16px)", opacity: 0, y: 40 }
//         : direction === "left"
//         ? { filter: "blur(16px)", opacity: 0, x: -40 }
//         : { filter: "blur(16px)", opacity: 0, x: 40 },
//     [direction]
//   );

//   const defaultTo = useMemo(
//     () => [
//       {
//         filter: "blur(8px)",
//         opacity: 0.6,
//         y: 0,
//         x: 0,
//         color: "#99f6e4", // teal glow (optional)
//       },
//       {
//         filter: "blur(0px)",
//         opacity: 1,
//         y: 0,
//         x: 0,
//         color: "#ffffff", // final color
//       },
//     ],
//     []
//   );

//   const fromSnapshot = animationFrom ?? defaultFrom;
//   const toSnapshots = animationTo ?? defaultTo;

//   const stepCount = toSnapshots.length + 1;
//   const totalDuration = stepDuration * (stepCount - 1);
//   const times = Array.from({ length: stepCount }, (_, i) =>
//     stepCount === 1 ? 0 : i / (stepCount - 1)
//   );

//   return (
//     <p
//       ref={ref}
//       className={className}
//       style={{
//         display: "flex",
//         flexWrap: "wrap",
//         gap: animateBy === "words" ? "0.5ch" : "0px",
//       }}
//     >
//       {elements.map((segment, index) => {
//         const animateKeyframes = buildKeyframes(fromSnapshot, toSnapshots);
//         const spanTransition = {
//           duration: totalDuration,
//           times,
//           delay: (index * delay) / 1000,
//           ease: easing,
//         };

//         return (
//           <motion.span
//             key={index}
//             className="inline-block will-change-[transform,filter,opacity]"
//             initial={fromSnapshot}
//             animate={inView ? animateKeyframes : fromSnapshot}
//             transition={spanTransition}
//             onAnimationComplete={
//               index === elements.length - 1 ? onAnimationComplete : undefined
//             }
//           >
//             {segment === " " ? "\u00A0" : segment}
//             {animateBy === "letters" && index < elements.length - 1 && "\u00A0"}
//           </motion.span>
//         );
//       })}
//     </p>
//   );
// };

// export default BlurText;
