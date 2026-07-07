import SectionHead from "./SectionHead";

export default function Section({
  id,
  sector,
  channel,
  title,
  lede,
  aside,
  children,
  className = "",
}: {
  id: string;
  sector: number;
  channel: string;
  title: React.ReactNode;
  lede?: React.ReactNode;
  aside?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      id={id}
      className={`section-anchor relative mx-auto w-full max-w-[1180px] px-5 py-24 md:px-8 md:py-32 ${className}`}
    >
      <SectionHead
        sector={sector}
        channel={channel}
        title={title}
        lede={lede}
        aside={aside}
      />
      {children}
    </section>
  );
}
