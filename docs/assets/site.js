(() => {
  const videos = Array.from(document.querySelectorAll("video[data-viewport-play]"));

  const getSources = (video) => Array.from(video.querySelectorAll("source[data-src]"));

  const loadVideo = (video) => {
    if (video.dataset.loaded === "true") return;

    const frame = video.closest(".video-frame");
    frame?.classList.remove("has-error");
    const status = frame?.querySelector(".video-status");
    if (status) status.textContent = "";

    getSources(video).forEach((source) => {
      source.src = source.dataset.src;
    });
    video.dataset.loaded = "true";
    video.load();
  };

  const unloadVideo = (video) => {
    if (video.dataset.loaded !== "true") return;

    video.pause();
    getSources(video).forEach((source) => source.removeAttribute("src"));
    video.load();
    delete video.dataset.loaded;
    video.closest(".video-frame")?.classList.remove("is-ready", "has-error");
  };

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
      if (video.dataset.loaded !== "true") return;
      const frame = video.closest(".video-frame");
      frame?.classList.add("has-error");
      const status = frame?.querySelector(".video-status");
      if (status) status.textContent = "Video unavailable";
    });
  });

  if ("IntersectionObserver" in window) {
    const loadObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) loadVideo(entry.target);
          else unloadVideo(entry.target);
        });
      },
      { threshold: 0, rootMargin: "400px 0px" },
    );

    const playObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const video = entry.target;
          if (entry.isIntersecting && entry.intersectionRatio >= 0.55) {
            loadVideo(video);
            video.play().catch(() => {});
          } else if (!video.paused) {
            video.pause();
          }
        });
      },
      { threshold: [0, 0.25, 0.55, 0.8, 1] },
    );

    videos.forEach((video) => {
      loadObserver.observe(video);
      playObserver.observe(video);
    });
  } else {
    videos.slice(0, 2).forEach(loadVideo);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pauseAll();
  });
})();
