(() => {
  const videos = Array.from(document.querySelectorAll("video[data-viewport-play]"));

  const pauseAll = () => {
    videos.forEach((video) => {
      if (!video.paused) video.pause();
    });
  };

  videos.forEach((video) => {
    video.addEventListener("loadedmetadata", () => {
      video.closest(".video-frame")?.classList.add("is-ready");
    });

    video.addEventListener("error", () => {
      const frame = video.closest(".video-frame");
      frame?.classList.add("has-error");
      const status = frame?.querySelector(".video-status");
      if (status) status.textContent = "Video unavailable";
    });
  });

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const video = entry.target;
          if (entry.isIntersecting && entry.intersectionRatio >= 0.55) {
            video.play().catch(() => {});
          } else if (!video.paused) {
            video.pause();
          }
        });
      },
      { threshold: [0, 0.25, 0.55, 0.8, 1], rootMargin: "80px 0px" },
    );

    videos.forEach((video) => observer.observe(video));
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pauseAll();
  });
})();
