import { useEffect, useState } from "react";
import "../styles.css";

const navigationLinks = [
  ["How it works", "#how"],
  ["Economics", "#economics"],
  ["Hardware", "#hardware"],
  ["Developers", "#developers"],
  ["Privacy & security", "#privacy"],
  ["Join early access", "#join"],
];

const workers = [
  { number: "01", name: "Worker 1", layers: "0–9", load: "68%", className: "worker-one" },
  { number: "02", name: "Worker 2", layers: "10–19", load: "72%", className: "worker-two" },
  { number: "03", name: "Worker 3", layers: "20–29", load: "65%", className: "worker-three" },
];

const sparklineHeights = [28, 42, 35, 58, 48, 72, 64, 84, 76, 92];

function Brand({ footer = false }) {
  return (
    <a className="brand" href="#" aria-label={footer ? "IntelHive home" : undefined}>
      <img className="brand-mark" src="/assets/intelhive-mark.svg" alt="" />
      <span>
        Intel<span className="brand-accent">Hive</span>
      </span>
    </a>
  );
}

function WorkerChip() {
  return (
    <svg className="chip-icon" viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <rect x="11" y="11" width="26" height="26" rx="5" stroke="currentColor" strokeWidth="2" />
      <path
        d="M18 7v4m6-4v4m6-4v4m-12 26v4m6-4v4m6-4v4M7 18h4m-4 6h4m-4 6h4m26-12h4m-4 6h4m-4 6h4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx="24" cy="24" r="6" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

function App() {
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const animatedElements = document.querySelectorAll(
      ".econ-card,.developer-panel,.owner-panel,.trust-grid div,.example-box",
    );
    if (!("IntersectionObserver" in window)) return undefined;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.style.opacity = "1";
            entry.target.style.transform = "translateY(0)";
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.08 },
    );

    animatedElements.forEach((element) => {
      element.style.opacity = "0";
      element.style.transform = "translateY(16px)";
      element.style.transition = "opacity .65s ease, transform .65s ease";
      observer.observe(element);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <>
      <div className="noise" />

      <header className="nav">
        <Brand />
        <nav className="nav-links" aria-label="Main navigation">
          {navigationLinks.slice(0, 5).map(([label, href]) => (
            <a href={href} key={href}>
              {label}
            </a>
          ))}
        </nav>
        <a className="nav-cta" href="#join">
          Join early access <span>→</span>
        </a>
        <button
          className="menu-btn"
          type="button"
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          aria-expanded={menuOpen}
          aria-controls="mobile-menu"
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? "×" : "☰"}
        </button>
        <nav
          className="mobile-menu"
          id="mobile-menu"
          aria-label="Mobile navigation"
          hidden={!menuOpen}
        >
          {navigationLinks.map(([label, href]) => (
            <a href={href} key={href} onClick={() => setMenuOpen(false)}>
              {label}
            </a>
          ))}
        </nav>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="pulse" />
              Collective intelligence, from the edge
            </div>
            <h1>
              Small devices.
              <br />
              <em>Collective intelligence.</em>
            </h1>
            <p className="hero-lead">
              IntelHive connects supported smartphones into a distributed AI network. Your device
              can contribute idle compute to AI workloads and receive rewards for useful work.
            </p>
            <div className="hero-actions">
              <a className="btn primary" href="#join">
                Join the network <span>→</span>
              </a>
              <a className="btn secondary" href="#economics">
                See how earning works
              </a>
            </div>
            <p className="fine-print">
              Earnings are variable and depend on demand, device performance, uptime, network
              conditions and program rules.
            </p>
          </div>

          <div className="phone-stage">
            <div className="orb orb-a" />
            <div className="orb orb-b" />
            <div className="grid-floor" />
            <div className="phone-card">
              <div className="phone-notch" />
              <div className="screen">
                <div className="screen-top">
                  <span>INTELHIVE NODE</span>
                  <span className="online">● ONLINE</span>
                </div>
                <div className="device-chip">NPU + GPU</div>
                <div className="earn-label">TODAY&apos;S COMPUTE</div>
                <div className="earn-value">
                  <span className="currency">₹</span> 184.20
                </div>
                <div className="sparkline">
                  {sparklineHeights.map((height, index) => (
                    <i key={index} style={{ height: `${height}%` }} />
                  ))}
                </div>
                <div className="screen-stats">
                  <div>
                    <b>76%</b>
                    <span>utilization</span>
                  </div>
                  <div>
                    <b>31°C</b>
                    <span>temperature</span>
                  </div>
                  <div>
                    <b>6.8h</b>
                    <span>active</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="floating-pill pill-1">
              AI inference <strong>+₹32.4</strong>
            </div>
            <div className="floating-pill pill-2">
              Model shard <strong>active</strong>
            </div>
            <div className="floating-pill pill-3">
              Network demand <strong>high</strong>
            </div>
          </div>
        </section>

        <section className="ticker" aria-label="How IntelHive works">
          <div>YOUR DEVICE</div>
          <span>→</span>
          <div>AI WORKLOAD</div>
          <span>→</span>
          <div>COMPUTE REWARD</div>
          <span>→</span>
          <div>YOUR WALLET</div>
        </section>

        <section className="section split" id="how">
          <div>
            <div className="section-kicker">01 / THE IDEA</div>
            <h2>
              Instead of letting a phone sit idle, <span>make its compute useful.</span>
            </h2>
          </div>
          <div className="section-copy">
            <p>
              Modern phones contain powerful GPUs and NPUs that spend much of their time
              underutilized. IntelHive is designed to coordinate those devices as a distributed AI
              compute network.
            </p>
            <p>
              For device owners, the idea is simple: acquire supported hardware, keep it connected
              under the network&apos;s operating rules, and receive rewards when your device completes
              useful workloads.
            </p>
          </div>
        </section>

        <section className="section economics" id="economics">
          <div className="section-kicker">02 / THE ECONOMICS</div>
          <div className="economics-head">
            <div>
              <h2>
                A phone can become <span>a small compute business.</span>
              </h2>
              <p>
                Think in terms of utilization and operating economics—not a guaranteed
                passive-income promise.
              </p>
            </div>
            <div className="formula">
              <span>Potential reward</span>
              <strong>work completed × network rate</strong>
              <small>minus electricity, connectivity &amp; device costs</small>
            </div>
          </div>

          <div className="economics-grid">
            <article className="econ-card">
              <span className="num">01</span>
              <h3>Choose hardware</h3>
              <p>
                Look for supported phones with capable GPU/NPU hardware, sufficient memory and good
                thermal characteristics.
              </p>
            </article>
            <article className="econ-card featured">
              <span className="num">02</span>
              <h3>Keep it available</h3>
              <p>
                When your device is online, cool and eligible, IntelHive can assign compatible AI
                work to its compute resources.
              </p>
            </article>
            <article className="econ-card">
              <span className="num">03</span>
              <h3>Receive rewards</h3>
              <p>
                Rewards can vary with workload demand, model type, performance, uptime, location and
                network conditions.
              </p>
            </article>
          </div>

          <div className="example-box">
            <div className="example-label">ILLUSTRATIVE EXAMPLE — NOT A RETURN PROMISE</div>
            <div className="example-content">
              <div>
                <span>Device purchase</span>
                <strong>₹30,000</strong>
              </div>
              <div>
                <span>Illustrative monthly compute rewards</span>
                <strong>₹3,000</strong>
              </div>
              <div>
                <span>Illustrative operating costs</span>
                <strong>− ₹700</strong>
              </div>
              <div className="total">
                <span>Illustrative net</span>
                <strong>₹2,300 / month</strong>
              </div>
            </div>
            <p>
              Actual economics may be substantially higher or lower. Network demand and device
              conditions determine real utilization.
            </p>
          </div>
        </section>

        <section className="section hardware" id="hardware">
          <div className="section-kicker">03 / BUILD YOUR NODE</div>
          <div className="hardware-head">
            <h2>
              One phone is a node.
              <br />
              <span>Thousands become infrastructure.</span>
            </h2>
            <p>Start with one device. Scale your compute footprint as the network grows.</p>
          </div>

          <div
            className="node-visual"
            role="group"
            aria-label="Illustration of one AI model divided across three connected phone workers"
          >
            <div className="network-heading">
              <span className="network-eyebrow">
                <span />
                DISTRIBUTED INFERENCE
              </span>
              <h3>
                One model. <span>Many phones.</span>
              </h3>
              <p>Work is split into model layers and processed across connected devices.</p>
            </div>
            <svg
              className="network-lines"
              viewBox="0 0 1000 500"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <path className="network-trunk" d="M500 184 V220 M170 220 H830" />
              <path className="network-branch branch-one" d="M170 220 V260" />
              <path className="network-branch branch-two" d="M500 220 V260" />
              <path className="network-branch branch-three" d="M830 220 V260" />
              <circle cx="170" cy="220" r="5" />
              <circle cx="500" cy="220" r="5" />
              <circle cx="830" cy="220" r="5" />
            </svg>
            <div className="model-node">
              <span className="model-icon" aria-hidden="true">
                <svg viewBox="0 0 48 48" fill="none">
                  <path
                    d="M24 7v9m0 0-12 8m12-8 12 8M12 24v5m24-5v5M24 16v15"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                  />
                  <circle cx="24" cy="7" r="4" />
                  <circle cx="12" cy="24" r="4" />
                  <circle cx="36" cy="24" r="4" />
                  <circle cx="24" cy="34" r="4" />
                </svg>
              </span>
              <span className="model-copy">
                <strong>AI MODEL</strong>
                <small>One workload · shared across the hive</small>
              </span>
              <span className="model-status">
                <i /> RUNNING
              </span>
            </div>
            <div className="workers" aria-label="Connected worker phones">
              {workers.map((worker) => (
                <article className={`worker ${worker.className}`} key={worker.number}>
                  <div className="worker-top">
                    <span className="worker-index">{worker.number}</span>
                    <span className="worker-live">
                      <i /> ONLINE
                    </span>
                  </div>
                  <div className="worker-phone">
                    <div className="worker-camera" />
                    <div className="worker-title">{worker.name}</div>
                    <div className="worker-layers">
                      Layers <b>{worker.layers}</b>
                      <small>~18B parameters</small>
                    </div>
                    <WorkerChip />
                    <div className="worker-load">
                      <span>GPU LOAD</span>
                      <strong>{worker.load}</strong>
                    </div>
                  </div>
                </article>
              ))}
            </div>
            <div className="network-foot">
              <span>
                <i /> 3 WORKERS CONNECTED
              </span>
              <span>ILLUSTRATIVE NETWORK VIEW</span>
            </div>
          </div>
        </section>

        <section className="section two-col" id="developers">
          <div className="developer-panel">
            <div className="section-kicker">FOR DEVELOPERS</div>
            <h2>Rent distributed mobile compute for AI.</h2>
            <p>
              Developers submit workloads to the network. IntelHive schedules eligible devices and,
              over time, can support model sharding so larger models are distributed across
              multiple phones.
            </p>
            <a className="text-link" href="#join">
              Build with the network <span>→</span>
            </a>
          </div>
          <div className="owner-panel">
            <div className="section-kicker">FOR DEVICE OWNERS</div>
            <h2>Know what your device is doing.</h2>
            <ul>
              <li>
                <span>→</span> Planned: clear workload and pause controls
              </li>
              <li>
                <span>→</span> Planned: battery and thermal limits
              </li>
              <li>
                <span>→</span> Planned: visibility into activity and rewards
              </li>
              <li>
                <span>→</span> Details to be documented before release
              </li>
            </ul>
            <a className="text-link" href="#privacy">
              Our privacy &amp; security approach <span>→</span>
            </a>
          </div>
        </section>

        <section className="section trust" id="privacy">
          <div className="section-kicker">04 / PRIVACY &amp; SECURITY</div>
          <div className="privacy-intro">
            <div>
              <h2>
                Your phone is personal.
                <br />
                <span>Trust has to be earned.</span>
              </h2>
            </div>
            <div className="privacy-note">
              <p>
                It is reasonable to ask what an app can access, what leaves your phone, and whether
                it can run without your say-so.
              </p>
              <p>
                <strong>
                  IntelHive is still a concept; there is no device app to install yet.
                </strong>{" "}
                We have not published a technical design or privacy policy, so we will not claim
                that personal data stays on-device or that the network is risk-free.
              </p>
            </div>
          </div>
          <div className="privacy-promise">
            <span className="privacy-symbol" aria-hidden="true">
              i
            </span>
            <p>
              Before asking anyone to install, we intend to publish plain-language answers to these
              questions and the technical details behind them.
            </p>
          </div>
          <div className="trust-grid">
            <div>
              <strong>Permissions &amp; access</strong>
              <span>
                Which phone permissions are requested, why each is needed, and what the app cannot
                access.
              </span>
            </div>
            <div>
              <strong>Workloads &amp; data</strong>
              <span>
                What data a job uses, whether it leaves your device, and how personal content is
                kept out of scope.
              </span>
            </div>
            <div>
              <strong>Security &amp; retention</strong>
              <span>
                How data is protected in transit and at rest, who can access it, and how long it is
                retained.
              </span>
            </div>
            <div>
              <strong>Your controls</strong>
              <span>
                How to see when compute is active, pause participation, revoke permissions, and
                remove the app.
              </span>
            </div>
          </div>
          <p className="privacy-footnote">
            These are questions IntelHive must answer before launch—not claims that a released app
            already provides these protections.
          </p>
        </section>

        <section className="cta" id="join">
          <div className="cta-glow" />
          <div className="section-kicker">EARLY ACCESS</div>
          <h2>
            Own the hardware.
            <br />
            <span>Power the AI network.</span>
          </h2>
          <p>
            Join the early-access list for node software, supported-device updates and network
            launch information.
          </p>
          <div className="waitlist-disabled" aria-describedby="waitlist-status">
            <span>Email signups aren&apos;t open yet</span>
            <button type="button" disabled>
              Coming soon
            </button>
          </div>
          <p className="form-message" id="waitlist-status" role="status">
            The waitlist is not connected yet. No email address is collected or stored.
          </p>
          <small>
            No investment guarantee. Rewards are subject to network availability, program terms and
            actual device performance.
          </small>
        </section>
      </main>

      <footer>
        <Brand footer />
        <p>Distributed AI compute, powered by phones.</p>
        <div className="footer-links">
          <a href="#how">How it works</a>
          <a href="#privacy">Privacy &amp; security</a>
          <a href="#join">Early access</a>
        </div>
        <span className="copyright">© 2026 IntelHive — concept site</span>
      </footer>
    </>
  );
}

export default App;
